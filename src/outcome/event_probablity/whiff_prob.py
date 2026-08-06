import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from src.pitch_value.event_prob.transformer.modules.data import (
    OutcomeProb,
    split_loaders,
)
from src.pitch_value.event_prob.transformer.modules.transformer import Transformer
from src.pitch_value.event_prob.transformer.modules.utils import (
    create_training_optimizers,
    lr_range_test_transf,
)
from torch import nn

device = "mps"


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

    labels = batch["whiff"].to(device)
    swing = batch["swing"].to(device)

    logits = model(pitchEmb, expEmb, pitcherID, batterID, atbat, count, lengths)
    mask = (labels != -1) & (swing == 1)
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
    compiled = torch.compile(model, dynamic=True)
    best_loss = np.inf

    for epoch in range(epochs):
        compiled.train()
        total_loss = 0.0

        for batch in train_loader:
            logits, labels, mask = forward(batch, compiled, device)
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

        compiled.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits, labels, mask = forward(batch, compiled, device)
                val_loss += loss_fn(logits[mask], labels[mask].float()).item()

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "weights/whiff_prob.pth")

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
            logits = torch.sigmoid(logits)
            if masking:
                all_preds.append(logits[mask])
                all_labels.append(labels[mask])
            else:
                all_preds.append(logits)
                all_labels.append(labels)

    return torch.cat(all_preds).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


if __name__ == "__main__":
    epochs = 100

    df = pl.read_parquet("data/cleaned_data/embed/output/fullBatter.parquet")
    df = df.with_columns(
        pitcher_year=pl.struct(["pitcher_id", "game_year"]).rank("dense") - 1,
        batter_year=pl.struct(["batter_id", "game_year"]).rank("dense") - 1,
    )

    pitchers = df["pitcher_year"].n_unique()
    batters = df["batter_year"].n_unique()
    pitches = df["pitch_group"].n_unique()

    data, train_loader, val_loader, test_loader = load_data(
        df, prefetch=4, workers=4, batch_size=4096
    )

    loss_fn = nn.BCEWithLogitsLoss()
    model = Transformer(
        dim=256,
        layers=8,
        dropout=0.1,
        nhead=8,
        output_dim=1,
        countdim=4,
        abdim=4,
        embed_dim=64,
        exp_dim=5,
        exp_proj_dim=16,
        pitcher_dim=64,
        batter_dim=64,
        n_pitcher_seasons=pitchers,
        n_batter_seasons=batters,
    )
    max_lr = lr_range_test_transf(model, train_loader, loss_fn, forward, device)

    total_steps = len(train_loader) * epochs
    muon_opti, adamw_opti, muon_sched, adamw_sched = create_training_optimizers(
        model,
        lr=max_lr,
        weight_decay=0.001,
        max_lr=max_lr,
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

    model.load_state_dict(torch.load("weights/whiff_prob.pth"))
    preds, labels = test(model, test_loader, masking=True)
    print(roc_auc_score(labels, preds))
    print(accuracy_score(labels, preds > 0.5))
    print(log_loss(labels, preds))
