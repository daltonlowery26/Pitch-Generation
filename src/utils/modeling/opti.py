from torch import nn
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import OneCycleLR


def tag(obj, *, no_wd=None, no_muon=None, lr_ratio=None):
    params = obj.parameters() if isinstance(obj, nn.Module) else [obj]
    for p in params:
        t = getattr(p, "_optim_tag", None)
        t = {"no_wd": False, "no_muon": False, "lr_ratio": 1.0} if t is None else t

        if no_wd is not None:
            t["no_wd"] = no_wd
        if no_muon is not None:
            t["no_muon"] = no_muon
        if lr_ratio is not None:
            t["lr_ratio"] = lr_ratio
        p._optim_tag = t

    return obj


def make_optimizers(
    model,
    *,
    lr,
    weight_decay=0.0,
    muon_momentum=0.90,
    adamw_betas=(0.9, 0.999),
    **one_cycle,
):
    muon, adamw = {}, {}
    seen = set()

    for module in model.modules():
        is_emb = isinstance(module, nn.Embedding)
        for p in module.parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            t = getattr(
                p, "_optim_tag", {"no_wd": False, "no_muon": False, "lr_ratio": 1.0}
            )
            use_muon = p.ndim == 2 and not is_emb and not t["no_muon"]
            wd = 0.0 if (t["no_wd"] or p.ndim < 2) else weight_decay
            (muon if use_muon else adamw).setdefault((t["lr_ratio"], wd), []).append(p)

    def groups(buckets, base_lr):
        return [
            {"params": ps, "lr": base_lr * r, "weight_decay": wd}
            for (r, wd), ps in buckets.items()
        ]

    def sched(opt, gs):
        return OneCycleLR(opt, max_lr=[g["lr"] for g in gs], **one_cycle)

    mg, ag = groups(muon, lr), groups(adamw, lr)
    muon_opt = Muon(mg, lr=lr, momentum=muon_momentum, adjust_lr_fn="match_rms_adamw")
    adamw_opt = AdamW(ag, lr=lr, betas=adamw_betas, weight_decay=weight_decay)

    return muon_opt, adamw_opt, sched(muon_opt, mg), sched(adamw_opt, ag)
