from datetime import date

import numpy as np
import polars as pl
import torch
from torch import nn

from src.outcome.stacked.modules.stacked_data import split_loaders
from src.outcome.stacked.modules.stacked_model import StackedCalibrator, brier_loss
from src.utils.modeling.config import load_config
from src.utils.modeling.opti import make_optimizers

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def forward_pass(model, batch, loss_fn):
    probs = batch["probs"].to(device, non_blocking=True)
    labels = {
        name: batch[name].to(device, non_blocking=True)
        for name in ("swing", "ball", "whiff", "foul", "bip")
    }
    return loss_fn(model(probs), labels)


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
        with torch.no_grad():
            for batch in valloader:
                val_loss += forward_pass(model, batch, lossfn)

        val_loss = val_loss.item() / len(valloader)
        print(f"epoch {epoch}, val {val_loss:3f}, train {train_loss:3f}", flush=True)

        if val_loss < best_loss:
            torch.save(
                {"model_state_dict": model.state_dict()},
                "weights/stack/stacked_calibrator",
            )
            best_loss = val_loss


if __name__ == "__main__":
    cfg = load_config("config/prob/stacked_calibrate.yml")
    df = pl.read_parquet("data/outcome/base_probs.parquet")
    train_loader, val_loader, _ = split_loaders(
        df=df,
        cutoff=date(2025, 7, 1),
        batch_size=cfg["training"]["batch_size"],
        prefetch=4,
        workers=4,
        pin=True,
    )

    model = StackedCalibrator(**cfg["arch"])
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    muon_opt, adamw_opt, muon_sched, adamw_sched = make_optimizers(
        model,
        lr=cfg["training"]["lr"],
        weight_decay=0.1,
        total_steps=total_steps,
        pct_start=1100 / total_steps,
    )

    train(
        model=model,
        trainloader=train_loader,
        valloader=val_loader,
        epochs=cfg["training"]["epochs"],
        lossfn=brier_loss,
        adam_opti=adamw_opt,
        muon_opti=muon_opt,
        adam_sched=adamw_sched,
        muon_sched=muon_sched,
    )
