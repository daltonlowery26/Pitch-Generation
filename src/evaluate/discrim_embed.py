from datetime import datetime

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.embeddings.pitch.modules.cvae import CVAE
from src.evaluate.modules.data import split_loaders
from src.evaluate.modules.discriminator import Discriminator
from src.utils.modeling.config import load_config
from src.utils.modeling.opti import make_optimizers

device = "cuda" if torch.cuda.is_available() else "mps"


def forward_pass(model, batch, loss_fn):
    embed = batch["embed"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)

    logits = model(embed)
    loss = loss_fn(logits, label)
    return loss


def train(
    model,
    trainloader,
    valloader,
    epochs,
    lossfn,
    adam_opti,
    muon_opti,
    adam_sched,
    muon_sched,
):
    model.to(device)
    c_model = torch.compile(model)

    best_loss = np.inf

    for epoch in range(epochs):
        train_loss = torch.zeros((), device=device)
        model.train()

        for batch in trainloader:
            loss = forward_pass(c_model, batch, lossfn)

            adam_opti.zero_grad()
            muon_opti.zero_grad()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            adam_opti.step()
            adam_sched.step()
            muon_opti.step()
            muon_sched.step()

            train_loss += loss.detach()

        train_loss = train_loss.item() / len(trainloader)
        val_loss = torch.zeros((), device=device)

        model.eval()
        for batch in valloader:
            loss = forward_pass(c_model, batch, lossfn)
            val_loss += loss

        val_loss = val_loss.item() / len(valloader)
        print(f"epoch {epoch}, val {val_loss:.6f}, train {train_loss:.6f}", flush=True)

        best_loss = min(best_loss, val_loss)


def evaluate(model, testloader):
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in testloader:
            embed = batch["embed"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            logits = model(embed)
            all_logits.append(logits.cpu())
            all_labels.append(label.cpu())

    logits = torch.cat(all_logits).squeeze(-1)
    labels = torch.cat(all_labels).squeeze(-1)

    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(np.float32)
    labels_np = labels.numpy()

    accuracy = (preds == labels_np).mean()
    auc = roc_auc_score(labels_np, probs)

    print(f"test accuracy: {accuracy:.4f}, test AUC: {auc:.4f}", flush=True)
    return accuracy, auc


if __name__ == "__main__":
    cfg = load_config("config/test/eval_embed.yml")

    df = pl.read_parquet(cfg["data_path"])

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
        "pitcher_id_year",
        "pitch_name",
    ]
    df = df.drop_nulls(subset=features)

    pitchers = df["pitcher_year"].n_unique()
    pitches = df["pitch_name"].n_unique()

    cvae_cfg = load_config("config/embed/cvae.yml")
    arch = cvae_cfg["arch"]
    cvae = CVAE(
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
    sd = torch.load(cfg["weight_path"], map_location=torch.device("cpu"))
    sd = sd.get("model_state", sd)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    cvae.load_state_dict(sd)

    testcutoff = datetime(2025, 7, 1).date()
    trainLoader, valLoader, testLoader = split_loaders(
        df=df,
        cvae_model=cvae,
        temperature=cfg["temperature"],
        cutoff=testcutoff,
        batch_size=cfg["training"]["batch_size"],
        prefetch=4,
        workers=4,
        drop_last=False,
    )

    model = Discriminator(
        inp=cfg["arch"]["inp"],
        dim=cfg["arch"]["dim"],
        blocks=cfg["arch"]["blocks"],
    )

    total_steps = len(trainLoader) * cfg["training"]["epochs"]
    muon_opt, adamw_opt, muon_sched, adamw_sched = make_optimizers(
        model,
        lr=cfg["training"]["lr"],
        weight_decay=0.1,
        total_steps=total_steps,
        pct_start=1100 / total_steps,
    )

    lossfn = nn.BCEWithLogitsLoss()

    train(
        model=model,
        trainloader=trainLoader,
        valloader=valLoader,
        epochs=cfg["training"]["epochs"],
        lossfn=lossfn,
        adam_opti=adamw_opt,
        muon_opti=muon_opt,
        adam_sched=adamw_sched,
        muon_sched=muon_sched,
    )

    evaluate(model, testLoader)
