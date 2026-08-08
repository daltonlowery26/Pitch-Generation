import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss
from torch import nn

from src.outcome.modules.seq_data import (
    OutcomeProb,
    split_loaders,
)
from src.outcome.modules.seq_transformer import Transformer
from src.utils.config import load_config
from src.utils.modeling.opti import make_optimizers

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def load_data(df, prefetch, workers, batch_size):
    data = OutcomeProb(df)
    return data, *split_loaders(data, df, batch_size, prefetch, workers, drop_last=True)


def forward(batch, model, device):
    pitchEmb = batch["pitchEmb"].to(device)
    pitcherID = batch["pitcherID"].to(device)
    batterID = batch["batterID"].to(device)
    expEmb = batch["expEmb"].to(device)

    lengths = batch["lengths"].to(device)
    atbat = batch["abNum"].to(device)
    count = batch["count"].to(device)

    labels = batch["bipLabel"].to(device)
    swing = batch["swing"].to(device)
    whiff = batch["whiff"].to(device)
    foul = batch["foul"].to(device)

    logits = model(pitchEmb, expEmb, pitcherID, batterID, atbat, count, lengths)
    mask = (labels[:, :, 0] != -1) & (swing == 1) & (whiff != 1) & (foul != 1)
    return logits, labels, mask


def train(
    model,
    train_loader,
    val_loader,
    muon_opti,
    adamw_opti,
    muon_sched,
    adamw_sched,
    loss_fn,
    epochs,
):
    model.to(device)
    best_loss = np.inf

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            logits, labels, mask = forward(batch, model, device)
            loss = loss_fn(logits[mask], labels[mask].float())

            muon_opti.zero_grad()
            adamw_opti.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            muon_opti.step()
            adamw_opti.step()
            muon_sched.step()
            adamw_sched.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits, labels, mask = forward(batch, model, device)
                val_loss += loss_fn(logits[mask], labels[mask].float()).item()

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "weights/bip_type.pth")

        print(
            f"epoch: {epoch} | train_loss: {train_loss:.4f} | val_loss: {avg_val_loss:.4f}"
        )


def test(model, test_loader, masking=True):
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, labels, mask = forward(batch, model, device)
            probs = torch.softmax(logits, dim=-1)
            if masking:
                all_preds.append(probs[mask])
                all_labels.append(labels[mask])
            else:
                all_preds.append(probs)
                all_labels.append(labels)

    return torch.cat(all_preds).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


if __name__ == "__main__":
    cfg = load_config("bip_type")
    epochs = cfg["training"]["epochs"]

    # preprocessing steps
    df = pl.read_parquet("data/player/expectation.parquet")

    # number of pitchers, batters, len
    pitchers = df["pitcher_year"].n_unique()
    batters = df["batter_year"].n_unique()
    pitches = df["pitch_group"].n_unique()

    data, train_loader, val_loader, test_loader = load_data(
        df, prefetch=4, workers=4, batch_size=cfg["training"]["batch_size"]
    )

    loss_fn = nn.CrossEntropyLoss()
    arch = cfg["arch"]
    model = Transformer(
        dim=arch["dim"],
        layers=arch["layers"],
        dropout=arch["dropout"],
        nhead=arch["nhead"],
        attn_fnn_block=arch["attn_ffn_blocks"],
        embed_dim=arch["embed_dim"],
        countdim=arch["countdim"],
        exp_dim=arch["exp_dim"],
        abdim=arch["abdim"],
        output_dim=5,
        exp_proj_dim=arch["exp_proj_dim"],
        batter_dim=arch["batter_dim"],
        n_pitcher_seasons=pitchers,
        n_batter_seasons=batters,
    )

    total_steps = len(train_loader) * epochs
    muon_opti, adamw_opti, muon_sched, adamw_sched = make_optimizers(
        model,
        lr=cfg["training"]["lr"],
        weight_decay=0.1,
        total_steps=total_steps,
        pct_start=1100 / total_steps,
    )

    train(
        model,
        train_loader,
        val_loader,
        muon_opti,
        adamw_opti,
        muon_sched,
        adamw_sched,
        loss_fn,
        epochs=epochs,
    )

    model.load_state_dict(torch.load("weights/bip_type.pth"))
    preds, labels = test(model, test_loader, masking=True)
    print(log_loss(labels, preds), flush=True)
