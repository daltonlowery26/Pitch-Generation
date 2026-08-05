import numpy as np
import polars as pl
import torch
import torch.nn as nn
from src.pitch_value.event_prob.stacked.modules.conditional import (
    conditional_loss,
    conditional_probs,
)
from src.pitch_value.event_prob.stacked.modules.data import (
    StackedDataset,
    split_loaders,
)
from src.pitch_value.event_prob.stacked.modules.utils import (
    create_training_optimizers,
    lr_range_test_stacked,
)

device = "mps"
WEIGHTS = "weights/calibmodel.pth"
EPOCHS = 30


def load_data(df, batch_size=4096):
    dataset = StackedDataset(df)
    return dataset, *split_loaders(dataset, df["game_date"].to_list(), batch_size)


def forward(batch, model, device):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    logits = model(x)
    mask = torch.ones(len(x), dtype=torch.bool, device=device)
    return logits, y, mask


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
            logits, y, mask = forward(batch, compiled, device)
            loss = loss_fn(logits, y)

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
                logits, y, _ = forward(batch, compiled, device)
                val_loss += loss_fn(logits, y).item()

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), WEIGHTS)

        print(
            f"epoch: {epoch} | train_loss: {train_loss:.4f} | val_loss: {avg_val_loss:.4f}"
        )


def test(model, test_loader):
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, y, _ = forward(batch, model, device)
            all_preds.append(conditional_probs(logits))
            all_labels.append(y)

    return torch.cat(all_preds).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def pred_df(model, df):
    dataset = StackedDataset(df)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False)
    model.to(device)
    model.eval()

    out = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            out.append(conditional_probs(logits).cpu().numpy())

    probs = np.concatenate(out, axis=0)
    return pl.concat(
        [
            df,
            pl.DataFrame(
                {
                    "swingProb_cal": probs[:, 0],
                    "whiffProb_cal": probs[:, 1],
                    "foulProb_cal": probs[:, 2],
                    "fbProb_cal": probs[:, 3],
                    "gbProb_cal": probs[:, 4],
                    "lineProb_cal": probs[:, 5],
                    "popProb_cal": probs[:, 6],
                }
            ),
        ],
        how="horizontal",
    )


def _sigmoid(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col >= 0)
        .then(1 / (1 + (-col).exp()))
        .otherwise(col.exp() / (1 + col.exp()))
    )


def main():
    df = pl.read_parquet("data/predections/fullEmbedPred.parquet")
    dataset, train_loader, val_loader, test_loader = load_data(df)

    model = CalibNet(dim=128)
    max_lr = lr_range_test_stacked(
        model, train_loader, conditional_loss, forward, device
    )

    total_steps = len(train_loader) * EPOCHS
    muon_opti, adamw_opti, muon_sched, adamw_sched = create_training_optimizers(
        model,
        lr=max_lr / 100,
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
        conditional_loss,
        epochs=EPOCHS,
    )

    model.load_state_dict(torch.load(WEIGHTS))
    preds, labels = test(model, test_loader)

    df = pred_df(model, df)

    df = df.with_columns(
        _sigmoid(pl.col("strikeProb")).fill_nan(0.5).alias("strikeProb"),
        _sigmoid(pl.col("hbpProbs")).fill_nan(0.015).alias("hbpProbs"),
    )

    df.write_parquet("data/predections/calibrated.parquet")


if __name__ == "__main__":
    main()
