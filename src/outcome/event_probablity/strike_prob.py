from datetime import datetime

import numpy as np
import polars as pl
import torch
from torch import nn

from src.outcome.modules.ffn.ffn_data import preprocess, split_loaders
from src.outcome.modules.ffn.ffn_model import Pitch_Predict
from src.utils.modeling.config import load_config
from src.utils.modeling.opti import make_optimizers

device = "cuda" if torch.cuda.is_available() else "mps"


def forward_pass(model, batch, loss_fn):
    # load data
    pitch = batch["pitch"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)

    # model pass & loss
    logits = model(pitch)
    loss = loss_fn(logits, label.reshape(-1, 1))

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

    best_loss = np.inf

    for epoch in range(epochs):
        train_loss = torch.zeros((), device=device)
        model.train()

        for batch in trainloader:
            loss = forward_pass(model, batch, lossfn)

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
            loss = forward_pass(model, batch, lossfn)
            val_loss += loss

        val_loss = val_loss.item() / len(valloader)
        print(f"epoch {epoch}, val {val_loss:3f}, train {train_loss:3f}", flush=True)

        if val_loss < best_loss:
            torch.save({"model_state_dict": model.state_dict()}, "weights/strike_prob")
            best_loss = val_loss


if __name__ == "__main__":
    cfg = load_config("config/prob/strike_prob.yml")

    df = pl.read_parquet("data/pitch/pitch_exp.parquet")
    df = preprocess(df, hbp=False)

    testcutoff = datetime(2025, 7, 1).date()
    trainLoader, valLoader, testLoader = split_loaders(
        df=df,
        label="ball",
        cutoff=testcutoff,
        batch_size=512,
        prefetch=4,
        workers=4,
        drop_last=False,
    )

    model = Pitch_Predict(
        inp=cfg["arch"]["inp"],
        out=cfg["arch"]["out"],
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
