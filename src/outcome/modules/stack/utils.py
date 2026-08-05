import numpy as np
import torch
import torch.nn as nn


def create_optimizers(model, lr, weight_decay, betas=(0.95, 0.999)):
    embedding_params = {
        id(p)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for p in module.parameters()
    }
    muon_params = [
        p for p in model.parameters() if p.ndim == 2 and id(p) not in embedding_params
    ]
    adamw_params = [
        p for p in model.parameters() if p.ndim != 2 or id(p) in embedding_params
    ]

    muon = torch.optim.Muon(
        muon_params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=betas[0],
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(
        adamw_params, lr=lr, weight_decay=weight_decay, betas=betas
    )
    return muon, adamw


def lr_range_test_stacked(
    model,
    train_loader,
    loss_fn,
    forward_fn,
    device,
    start_lr=1e-7,
    end_lr=10.0,
    num_iter=100,
    smooth_f=0.05,
):
    model.to(device)
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    muon, adamw = create_optimizers(model, lr=start_lr, weight_decay=0.001)
    gamma = (end_lr / start_lr) ** (1 / num_iter)
    muon_sched = torch.optim.lr_scheduler.ExponentialLR(muon, gamma=gamma)
    adamw_sched = torch.optim.lr_scheduler.ExponentialLR(adamw, gamma=gamma)

    lrs, losses = [], []
    smoothed_loss = 0.0
    best_loss = np.inf
    loader_iter = iter(train_loader)

    model.train()
    for i in range(num_iter):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        logits, labels, mask = forward_fn(batch, model, device)
        loss = loss_fn(logits[mask], labels[mask])

        muon.zero_grad()
        adamw.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        muon.step()
        adamw.step()

        lr = adamw.param_groups[0]["lr"]
        raw = loss.item()
        smoothed_loss = (
            smooth_f * raw + (1 - smooth_f) * smoothed_loss if i > 0 else raw
        )
        debias = smoothed_loss / (1 - (1 - smooth_f) ** (i + 1))

        lrs.append(lr)
        losses.append(debias)

        if debias < best_loss:
            best_loss = debias
        elif debias > 4 * best_loss:
            break

        muon_sched.step()
        adamw_sched.step()

    model.load_state_dict(init_state)

    gradients = np.gradient(np.array(losses))
    max_lr = lrs[np.argmin(gradients)] / 10

    print(f"LR range test complete | suggested max_lr: {max_lr:.2e}")
    return max_lr


# general param tagging
def tag_params(model, lr, groups, default_weight_decay=0.0):
    tagged = {}
    for name, param in model.named_parameters():
        ratio, weight_decay = 1.0, default_weight_decay
        for key, (r, wd) in groups.items():
            if key in name:
                ratio, weight_decay = r, wd
                break
        tagged.setdefault((ratio, weight_decay), []).append(param)

    return [
        {"params": params, "lr": lr * ratio, "weight_decay": weight_decay}
        for (ratio, weight_decay), params in tagged.items()
    ]
