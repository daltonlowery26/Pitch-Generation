# batter expectation of pitch in different counts
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.embeddings.expectation.modules.data import collate_fn, load_usage_data
from src.embeddings.expectation.modules.transformer import BatterExp
from src.embeddings.expectation.modules.utils import (
    get,
    loadBatch,
    multistep_loss,
)
from src.utils.config import load_config
from src.utils.modeling.opti import make_optimizers

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def forwardLoss(model, batch, repMask, i1, aux_h, aux_w, h1_w, aux=True):
    logits, _ = model(
        batch["embeds"],
        batch["atbat"],
        batch["count"],
        batch["pitcherID"],
        batch["batterID"],
        repMask,
        batch["lengths"],
    )

    # 1-ahead, position aligned with first prior
    prior_logits = model.get_prior(
        batch["pitcherID"][:, 0],
        batch["batterID"][:, 0],
        repMask,
    )
    full_h1 = torch.cat([prior_logits.unsqueeze(1), logits[:, i1, :-1]], dim=1)
    h1_loss = F.cross_entropy(
        full_h1.reshape(-1, full_h1.size(-1)),
        batch["labels"].reshape(-1),
        ignore_index=-1,
    )

    # val only one head
    if not aux:
        return h1_loss, h1_loss

    aux_loss, _ = multistep_loss(
        logits[:, 1:], batch["labels"], batch["lengths"], aux_h, weights=aux_w
    )
    loss = h1_w * h1_loss + aux_loss
    return loss, h1_loss


def usageTrain(
    model,
    dataLoader,
    valLoader,
    muon_opti,
    adamw_opti,
    muon_sched,
    adamw_sched,
    repMask,
    epochs,
    h1_w,
    aux_w,
):
    best_loss = np.inf
    i1 = model.horizons.index(1)
    aux_h = model.horizons[1:]

    for epoch in range(epochs):
        model.train()
        total_loss, h_1_loss = (
            torch.zeros((), device=device),
            torch.zeros((), device=device),
        )
        for batchFeat in dataLoader:
            batch = loadBatch(batchFeat, device)
            muon_opti.zero_grad(set_to_none=True)
            adamw_opti.zero_grad(set_to_none=True)

            loss, h1_loss = forwardLoss(
                model, batch, repMask, i1, aux_h, aux_w, h1_w, aux=True
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            muon_opti.step()
            adamw_opti.step()
            muon_sched.step()
            adamw_sched.step()

            total_loss += loss.detach()
            h_1_loss += h1_loss.detach()

        train_loss = total_loss.item() / len(dataLoader)
        train_1 = h_1_loss.item() / len(dataLoader)

        val_loss = torch.zeros((), device=device)
        with torch.inference_mode():
            for valFeat in valLoader:
                batch = loadBatch(valFeat, device)
                _, h1 = forwardLoss(
                    model, batch, repMask, i1, aux_h, aux_w, h1_w, aux=False
                )
                val_loss += h1.detach()
        avg_val_loss = val_loss.item() / len(valLoader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "muon_state": muon_opti.state_dict(),
                    "adamw_state": adamw_opti.state_dict(),
                },
                "weights/usage.pth",
            )
        print(
            f"epoch: {epoch} | oneahead loss {train_1:.3f}| "
            f"train_loss: {train_loss:.3f} | val_loss: {avg_val_loss:.3f}"
        )

    return best_loss


def usageTest(model, test_loader, repMask, device=device):
    preds, labels = get(model, test_loader, repMask, device)
    y = labels
    yhat = preds.argmax(axis=1)
    C = model.out_dim

    return {
        "f1": f1_score(y, yhat, average="macro"),
        "roc_auc": roc_auc_score(
            y, preds, multi_class="ovr", average="macro", labels=list(range(C))
        ),
        "log_loss": log_loss(y, preds, labels=list(range(C))),
        "accuracy": accuracy_score(y, yhat),
    }


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cfg = load_config("expectation")
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]

    val_start = datetime(2022, 8, 1).date()
    val_end = datetime(2022, 9, 30).date()
    test_cutoff = datetime(2025, 7, 1).date()

    data = load_usage_data(
        "data/player/player.parquet",
        val_start,
        val_end,
        test_cutoff,
    )
    pitchers, batters, pitches, max_len = (
        data.pitchers,
        data.batters,
        data.pitches,
        data.max_len,
    )
    mask = data.mask.to(device)

    # data loaders
    train_loader = DataLoader(
        data.train_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        data.val_set,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        data.test_set,
        batch_size=batch_size,
        num_workers=4,
        drop_last=False,
        prefetch_factor=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # model set up
    loss_fn = nn.CrossEntropyLoss()

    arch = cfg["arch"]
    model = BatterExp(
        dim=arch["dim"],
        out_dim=pitches,
        nheads=arch["nheads"],
        layers=arch["layers"],
        dropout=arch["dropout"],
        max_len=max_len,
        horizons=tuple(arch["horizons"]),
        pitcher_dim=arch["pitcher_dim"],
        batter_dim=arch["batter_dim"],
        embed_dim=arch["embed_dim"],
        batter_seasons=batters,
        pitcher_seasons=pitchers,
    )
    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # one cycle
    total_steps = len(train_loader) * epochs
    muon_opti, adamw_opti, muon_sched, adamw_sched = make_optimizers(
        model,
        lr=cfg["training"]["lr"],
        weight_decay=0.1,
        total_steps=total_steps,
        pct_start=1100 / total_steps,
        anneal_strategy="cos",
        div_factor=100,
        final_div_factor=1e2,
    )

    # train loop
    h1_w = 0.3
    aux_w = (0.25, 0.25, 0.2)

    usageTrain(
        model=model,
        dataLoader=train_loader,
        valLoader=val_loader,
        muon_opti=muon_opti,
        adamw_opti=adamw_opti,
        muon_sched=muon_sched,
        adamw_sched=adamw_sched,
        repMask=mask,
        epochs=epochs,
        h1_w=h1_w,
        aux_w=aux_w,
    )

    # test, on the best val checkpoint rather than the final epoch
    ckpt = torch.load("weights/usage.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"testing epoch {ckpt['epoch']} weights", flush=True)
    print(usageTest(model, test_loader, mask, device), flush=True)
