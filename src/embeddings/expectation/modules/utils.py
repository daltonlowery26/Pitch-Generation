import torch
import torch.nn.functional as F


def loadBatch(batchFeat, device):
    return {
        "embeds": batchFeat["embeds"].to(device, non_blocking=True),
        "labels": batchFeat["labels"].to(device, non_blocking=True).long(),
        "lengths": batchFeat["lengths"].to(device, non_blocking=True),
        "atbat": batchFeat["abNum"].to(device, non_blocking=True),
        "count": batchFeat["count"].to(device, non_blocking=True),
        "pitcherID": batchFeat["pitcherID"].to(device, non_blocking=True),
        "batterID": batchFeat["batterID"].to(device, non_blocking=True),
    }

# general inference forward pass
def inferForward(model, batch, repMask, device):
    batch = loadBatch(batch, device)
    logits, rep = model(
        batch["embeds"],
        batch["atbat"],
        batch["count"],
        batch["pitcherID"],
        batch["batterID"],
        repMask,
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

# return labels and preds for testing
def get(model, loader, repMask, device):
    labels = []
    preds = []
    for batch in loader:
        _, _, probs, _, prior_probs = inferForward(model, batch, repMask, device)

        full_probs = torch.cat([prior_probs.unsqueeze(1), probs[:, :-1]], dim=1) 

        flat_probs = full_probs.reshape(-1, full_probs.size(-1))
        flat_labels = batch['labels'].reshape(-1)
        
        mask = flat_labels != -1
        labels.append(flat_labels[mask].cpu().numpy())
        preds.append(flat_probs[mask].cpu().numpy())
        
    return (
        np.concatenate(preds),
        np.concatenate(labels),
    )
        

# masking out what pitchers actually throw
def repertoire_mask(df, num_pitchers, num_pitches):
    df = df.sort(["pitcher_year"])
    mask = torch.zeros(num_pitchers, num_pitches, dtype=torch.bool)
    for row in df.select(["pitcher_year", "pitch_group"]).unique().iter_rows():
        # pitcher, pitch, set mask to true
        mask[row[0], row[1]] = True
    return mask


# multistep loss, consider times t + 1 + 2 + 3
def multistep_loss(logits, labels, lengths, horizons, weights, ignore_index=-1):
    B, _, T, C = logits.shape

    # check if pos is within valid range
    pos = torch.arange(T, device=logits.device)

    # weighted loss
    total = logits.new_zeros(())
    
    parts = {}
    for i, k in enumerate(horizons):
        # shift target at this step to first step in seq
        tgt = torch.cat([labels[:, k:], labels.new_full((B, k), ignore_index)], dim=1)

        # only keep valid postions beyond length
        valid = (pos[None, :] + k) < lengths[:, None]
        tgt = tgt.masked_fill(~valid, ignore_index)

        # reshape labels and targets for cross entropy
        li = logits[:, i].reshape(B * T, C)
        ti = tgt.reshape(B * T)

        # comput loss if valid indexes
        if (ti != ignore_index).any():
            loss_k = F.cross_entropy(li, ti, ignore_index=ignore_index)
        else:
            loss_k = logits.new_zeros(())

        # add weighted portion
        w = weights[i]
        total = total + w * loss_k
        parts[f"h{k}"] = loss_k.detach()
        
    return total, parts