# %% package
import gc
import math
import os
from datetime import date

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch import nn

from src.embeddings.pitch.modules.cvae import CVAE, cvae_loss
from src.embeddings.pitch.modules.umap import UmapDataset, UmapLoss, umapGraph
from src.utils.config import load_config
from src.utils.modeling.opti import make_optimizers

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


# get batch of index, randomly permutate
def indexs(heads, tails, batch_size, device, gen, shuffle=True):
    n = heads.numel()

    if shuffle:
        perm = torch.randperm(n, device=device, generator=gen)
    else:
        perm = torch.arange(n, device=device)

    n_full = (n // batch_size) * batch_size
    perm = perm[:n_full].view(-1, batch_size)
    for row in perm:
        yield heads[row], tails[row]


@torch._disable_dynamo
def get_batch(kmu, h_index, t_index):
    x_h = kmu[h_index, :10]
    x_t = kmu[t_index, :10]
    p_h, p_t = kmu[h_index, 10].long(), kmu[t_index, 10].long()
    pi_h, pi_t = kmu[h_index, 11].long(), kmu[t_index, 11].long()
    return x_h, x_t, p_h, p_t, pi_h, pi_t


@torch.compile
def forward(model, kmu, h_index, t_index, reconLoss, umap_loss):
    x_h, x_t, p_h, p_t, pi_h, pi_t = get_batch(kmu, h_index, t_index)

    (
        recon_h,
        mu_qh,
        logvar_qh,
    ) = model(x_h, p_h, pi_h)
    (
        recon_t,
        mu_qt,
        logvar_qt,
    ) = model(x_t, p_t, pi_t)

    umapL = umap_loss(mu_qh, mu_qt)
    loss_h = cvae_loss(recon_h, x_h, mu_qh, logvar_qh, reconLoss, beta=0.4)
    loss_t = cvae_loss(recon_t, x_t, mu_qt, logvar_qt, reconLoss, beta=0.4)
    return umapL + loss_h + loss_t


def train(
    model,
    trainDataset,
    valDataset,
    kmuTrain,
    kmuVal,
    muon,
    adamw,
    muon_sched,
    adamw_sched,
    umap_loss,
    reconLoss,
    batchsize,
    epochs,
):
    c_model = torch.compile(model)
    c_model.to(device)

    best_loss = np.inf
    gen = torch.Generator(device=device).manual_seed(26)

    # features stay on GPU
    kmu_train = torch.tensor(kmuTrain["kmu"].to_list(), device=device)
    kmu_val = torch.tensor(kmuVal["kmu"].to_list(), device=device)

    val_heads = torch.as_tensor(valDataset.heads, device=device)
    val_tails = torch.as_tensor(valDataset.tails, device=device)
    n_val_batches = val_heads.numel() // batchsize

    for epoch in range(epochs):
        trainDataset.resample()
        train_heads = torch.as_tensor(trainDataset.heads, device=device)
        train_tails = torch.as_tensor(trainDataset.tails, device=device)
        n_train_batches = train_heads.numel() // batchsize

        # train
        total_loss = 0.0
        model.train()
        for h_index, t_index in indexs(
            train_heads, train_tails, batchsize, device, gen, shuffle=True
        ):
            muon.zero_grad()
            adamw.zero_grad()
            loss = forward(
                c_model,
                kmu_train,
                h_index,
                t_index,
                reconLoss=reconLoss,
                umap_loss=umap_loss,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            muon.step()
            adamw.step()
            muon_sched.step()
            adamw_sched.step()

            total_loss += loss.detach()

        train_loss = (total_loss / n_train_batches).item()

        # validation
        val_loss = 0.0
        model.eval()
        with torch.inference_mode():
            for h_index, t_index in indexs(
                val_heads, val_tails, batchsize, device, gen, shuffle=False
            ):
                val_loss += forward(
                    c_model,
                    kmu_val,
                    h_index,
                    t_index,
                    reconLoss=reconLoss,
                    umap_loss=umap_loss,
                ).detach()

        avg_val_loss = (val_loss / n_val_batches).item()

        # checkpoint best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint_path = "weights/cvae_map.pth"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                checkpoint_path,
            )

        print(
            f"epoch: {epoch}, valLoss: {avg_val_loss}, trainLoss: {train_loss}",
            flush=True,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cfg = load_config("cvae")
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]

    # %% input data consruction
    input = pl.read_parquet(
        "/domino/home/jdlower/scratch/data/source/pitch_2015_2026.parquet"
    )

    features = [
        "hra",
        "vra",
        "release_height",
        "release_x",
        "ay",
        "az",
        "ax",
        "vx0",
        "vy0",
        "vz0",
        "pitcher_year",
        "pitch_name",
    ]
    input = input.drop_nulls(subset=features)
    print(len(input), flush=True)

    # count, try to catch embed errors
    pitchers = input["pitcher_year"].n_unique()
    pitches = input["pitch_name"].n_unique()
    assert input["pitch_name"].max() == pitches - 1
    assert input["pitcher_year"].max() == pitchers - 1

    # normalize
    input = input.with_columns(
        kmu=pl.concat_list(
            [(pl.col(f) - pl.col(f).mean()) / pl.col(f).std() for f in features[:10]]
            + [pl.col(f) for f in features[10:]]
        )
    )

    # presplit data
    val_start = date(2025, 7, 1)
    trainD = input.filter(pl.col("game_date") < val_start)
    valD = input.filter(pl.col("game_date") >= val_start)

    # umapGraph, split correctly
    graph_dir = "/domino/home/jdlower/scratch/data/graphs"
    train_graph_path = f"{graph_dir}/train100.npz"
    val_graph_path = f"{graph_dir}/val100.npz"

    if not (os.path.exists(train_graph_path) and os.path.exists(val_graph_path)):
        sp.save_npz(train_graph_path, umapGraph(trainD, n_neighbors=100))
        sp.save_npz(val_graph_path, umapGraph(valD, n_neighbors=100))

    trainGraph = sp.load_npz(train_graph_path)
    valGraph = sp.load_npz(val_graph_path)

    # datasets
    trainDataset = UmapDataset(trainGraph)
    valDataset = UmapDataset(valGraph)

    # %% model
    trainDataset.resample()
    valDataset.resample()
    steps_per_epoch = math.ceil(len(trainDataset) / batch_size)
    total_steps = steps_per_epoch * epochs

    arch = cfg["arch"]
    model = CVAE(
        input_dim=arch["input_dim"],
        pitch_dim=arch["pitch_dim"],
        pitcher_dim=arch["pitcher_dim"],
        h_dim=arch["h_dim"],
        latent_dim=arch["latent_dim"],
        dropout=arch["dropout"],
        n_layers=arch["n_layers"],
        n_pitchers=pitchers,
        n_pitches=pitches,
    )
    print(sum(p.numel() for p in model.parameters()), flush=True)

    # loss functions
    reconLoss = nn.MSELoss()
    umapLoss = UmapLoss(minDist=0.3, spread=1.0, negSample=20)

    # optimizer and lr sched
    muon, adamw, onecycle_muon, onecycle_adamw = make_optimizers(
        model,
        lr=cfg["training"]["lr"],
        weight_decay=0.1,
        total_steps=total_steps,
        pct_start=1100 / total_steps,
        anneal_strategy="cos",
        div_factor=100,
        final_div_factor=1e2,
    )

    # %% train model
    gc.collect()
    torch.cuda.empty_cache()

    train(
        model=model,
        trainDataset=trainDataset,
        valDataset=valDataset,
        kmuTrain=trainD,
        kmuVal=valD,
        muon=muon,
        adamw=adamw,
        muon_sched=onecycle_muon,
        adamw_sched=onecycle_adamw,
        umap_loss=umapLoss,
        reconLoss=reconLoss,
        batchsize=batch_size,
        epochs=epochs,
    )
