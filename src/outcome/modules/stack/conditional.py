import torch
import torch.nn.functional as F


def conditional_loss(logits, y):
    """
    Joint conditional chain loss over the 7 outcome logits:
      [swing, whiff, foul, fly_ball, ground_ball, line_drive, popup]
    """
    z_sw, z_wh, z_fo = logits[:, 0], logits[:, 1], logits[:, 2]
    z_bip = logits[:, 3:7]

    y_sw, y_wh, y_fo = y[:, 0], y[:, 1], y[:, 2]
    y_bip = y[:, 3:7]
    reached_bip = y_sw * (1 - y_wh) * (1 - y_fo)

    L_sw = F.binary_cross_entropy_with_logits(z_sw, y_sw, reduction="none")
    L_wh = F.binary_cross_entropy_with_logits(z_wh, y_wh, reduction="none") * y_sw
    L_fo = (
        F.binary_cross_entropy_with_logits(z_fo, y_fo, reduction="none")
        * y_sw
        * (1 - y_wh)
    )
    L_bip = -(y_bip * F.log_softmax(z_bip, dim=-1)).sum(dim=-1) * reached_bip

    return (L_sw + L_wh + L_fo + L_bip).mean()


def conditional_probs(logits):
    """
    Convert calibrated logits to joint conditional probabilities.
    Returns (N, 7): [p_swing, p_whiff, p_foul, p_fb, p_gb, p_line, p_pop]
    """
    p_sw = torch.sigmoid(logits[:, 0])
    p_wh = torch.sigmoid(logits[:, 1])
    p_fo = torch.sigmoid(logits[:, 2])
    p_bip = F.softmax(logits[:, 3:7], dim=-1)

    p_whiff = p_sw * p_wh
    p_foul = p_sw * (1 - p_wh) * p_fo
    p_bip_joint = (p_sw * (1 - p_wh) * (1 - p_fo)).unsqueeze(-1) * p_bip

    return torch.stack(
        [
            p_sw,
            p_whiff,
            p_foul,
            p_bip_joint[:, 0],
            p_bip_joint[:, 1],
            p_bip_joint[:, 2],
            p_bip_joint[:, 3],
        ],
        dim=-1,
    )
