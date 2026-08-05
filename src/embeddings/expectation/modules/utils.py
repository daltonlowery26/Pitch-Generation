import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inferForward(model, batch, repMask):
    logits, rep = model(
        batch["embeds"],
        batch["atbat"],
        batch["count"],
        batch["pitcherID"],
        batch["batterID"],
        batch["lengths"],
    )
    i1 = model.horizons.index(1)
    logits = logits[:, i1]
    probs = torch.softmax(logits, dim=-1)
    prior_logits = model.get_prior(
        batch["pitcherID"][:, 0],
        batch["batterID"][:, 0],
        repMask,
    )
    prior_probs = torch.softmax(prior_logits, dim=-1)
    return logits, rep, probs, prior_logits, prior_probs


def get(model, _loadbatch, _forward, loader, repMask, device):
    fullReps, fullLogits, fullPreds, fullLabels = [], [], [], []
    model.eval()
    repMask = repMask.to(device)

    with torch.no_grad():
        for feat in loader:
            batch = _loadbatch(feat, device)
            logits, rep, probs, prior_logits, prior_probs = inferForward(
                model, batch, repMask
            )

            lengths, labels = batch["lengths"], batch["labels"]
            # strip padding per sequence
            for i in range(batch["embeds"].shape[0]):
                L = lengths[i].item()
                fullLogits.append(
                    torch.cat([prior_logits[i : i + 1], logits[i, : L - 1]], dim=0)
                    .cpu()
                    .numpy()
                )
                fullPreds.append(
                    torch.cat([prior_probs[i : i + 1], probs[i, : L - 1]], dim=0)
                    .cpu()
                    .numpy()
                )
                fullLabels.append(labels[i, :L].cpu().numpy())
                fullReps.append(rep[i, :L].cpu().numpy())

    fullReps = np.concatenate(fullReps, axis=0)
    fullLogits = np.concatenate(fullLogits, axis=0)
    fullPreds = np.concatenate(fullPreds, axis=0)
    fullLabels = np.concatenate(fullLabels)
    return fullLogits, fullReps, fullPreds, fullLabels


# masking out what pitchers actually throw
def repertoire_mask(df, num_pitchers, num_pitches):
    df = df.sort(["pitcher_year"])
    mask = torch.zeros(num_pitchers, num_pitches, dtype=torch.bool)
    for row in df.select(["pitcher_year", "pitch_group"]).unique().iter_rows():
        # pitcher, pitch, set mask to true
        mask[row[0], row[1]] = True
    return mask


# multistep loss, consider times t + 1 + 2 + 3
def multistep_loss(
    logits, labels, lengths, horizons, weights=[0.5, 0.33, 0.17], ignore_index=-1
):
    B, _, T, C = logits.shape
    pos = torch.arange(T, device=logits.device)
    total = logits.new_zeros(())
    parts = {}
    for i, k in enumerate(horizons):
        tgt = torch.cat([labels[:, k:], labels.new_full((B, k), ignore_index)], dim=1)
        valid = (pos[None, :] + k) < lengths[:, None]
        tgt = tgt.masked_fill(~valid, ignore_index)
        li = logits[:, i].reshape(B * T, C)
        ti = tgt.reshape(B * T)
        if (ti != ignore_index).any():
            loss_k = F.cross_entropy(li, ti, ignore_index=ignore_index)
        else:
            loss_k = logits.new_zeros(())
        w = 1.0 if weights is None else weights[i]
        total = total + w * loss_k
        parts[f"h{k}"] = loss_k.detach()
    return total, parts


def lr_range_test(
    model,
    loader,
    muon_opti,
    adamw_opti,
    rep_mask,
    _forwardpass,
    _loadbatch,
    device,
    start_lr=1e-7,
    end_lr=1.0,
    num_steps=200,
):
    lrs, losses = [], []

    muon_state = copy.deepcopy(muon_opti.state_dict())
    adamw_state = copy.deepcopy(adamw_opti.state_dict())
    model_state = copy.deepcopy(model.state_dict())

    gamma = (end_lr / start_lr) ** (1 / num_steps)
    muon_sched = torch.optim.lr_scheduler.ExponentialLR(muon_opti, gamma=gamma)
    adamw_sched = torch.optim.lr_scheduler.ExponentialLR(adamw_opti, gamma=gamma)
    for opti in (muon_opti, adamw_opti):
        for pg in opti.param_groups:
            pg["lr"] = start_lr

    i1 = model.horizons.index(1)
    aux_h = model.horizons[1:]
    aux_w = [0.30, 0.15, 0.15]
    h1_w = 0.4

    model.train()
    model.to(device)
    loader_iter = iter(loader)
    smooth_loss, beta = None, 0.98

    for step in range(num_steps):
        try:
            batchFeat = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batchFeat = next(loader_iter)

        batch = _loadbatch(batchFeat, device)

        muon_opti.zero_grad(set_to_none=True)
        adamw_opti.zero_grad(set_to_none=True)
        loss, _ = _forwardpass(model, batch, rep_mask, i1, aux_h, aux_w, h1_w, aux=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        muon_opti.step()
        adamw_opti.step()
        muon_sched.step()
        adamw_sched.step()

        # ema smoothed loss
        raw = loss.item()
        smooth_loss = (
            raw if smooth_loss is None else beta * smooth_loss + (1 - beta) * raw
        )
        lrs.append(adamw_sched.get_last_lr()[0])
        losses.append(smooth_loss)
        if smooth_loss > 10 * losses[0]:
            break

    model.load_state_dict(model_state)
    muon_opti.load_state_dict(muon_state)
    adamw_opti.load_state_dict(adamw_state)

    return lrs, losses
