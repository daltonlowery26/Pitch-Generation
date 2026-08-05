
import torch
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


# masking out what pitchers actually throw
def repertoire_mask(df, num_pitchers, num_pitches):
    df = df.sort(["pitcher_year"])
    mask = torch.zeros(num_pitchers, num_pitches, dtype=torch.bool)
    for row in df.select(["pitcher_year", "pitch_group"]).unique().iter_rows():
        # pitcher, pitch, set mask to true
        mask[row[0], row[1]] = True
    return mask


# multistep loss, consider times t + 1 + 2 + 3
def multistep_loss(logits, labels, lengths, horizons, weights=[0.5, 0.33, 0.17], ignore_index=-1):
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