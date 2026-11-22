from itertools import permutations

import torch


def pairwise_si_sdr(estimate, target, lengths=None):
    """Return [B, estimated source, reference source] scores in dB."""
    if estimate.shape != target.shape or estimate.ndim != 3:
        raise ValueError("Expected matching [batch, sources, time] tensors")
    estimate, target = estimate.float(), target.float()
    batch, _, time = target.shape
    if lengths is None:
        lengths = torch.full((batch,), time, device=target.device, dtype=torch.long)
    if lengths.shape != (batch,) or (lengths <= 0).any() or (lengths > time).any():
        raise ValueError("Invalid valid lengths")
    mask = (torch.arange(time, device=target.device)[None, None] < lengths[:, None, None]).float()
    count = lengths[:, None, None]
    estimate = (estimate - (estimate * mask).sum(-1, keepdim=True) / count) * mask
    target = (target - (target * mask).sum(-1, keepdim=True) / count) * mask
    energy = target.square().sum(-1)
    if (energy <= 1e-8).any():
        raise ValueError("SI-SDR requires non-silent reference sources")
    dot = torch.einsum("bit,bjt->bij", estimate, target)
    projection_energy = dot.square() / energy[:, None, :]
    projection = dot[..., None] / energy[:, None, :, None] * target[:, None]
    noise_energy = (estimate[:, :, None] - projection).square().sum(-1)
    return 10 * torch.log10(projection_energy / (noise_energy + 1e-8) + 1e-8)


def pit_si_sdr(estimate, target, lengths=None):
    scores = pairwise_si_sdr(estimate, target, lengths)
    count = target.shape[1]
    orders = list(permutations(range(count)))
    candidates = torch.stack([scores[:, range(count), order].mean(-1) for order in orders], -1)
    best, indices = candidates.max(-1)
    return best, torch.tensor(orders, device=target.device)[indices]


def pit_objective(estimate, target, lengths=None, scale_weight=0.0, *, return_assignment=False):
    """Joint PIT assignment for SI-SDR and scale-sensitive normalized waveform error."""
    if not 0 <= scale_weight < float("inf"):
        raise ValueError("scale_weight must be finite and nonnegative")
    scores = pairwise_si_sdr(estimate, target, lengths)
    estimate, target = estimate.float(), target.float()
    time = target.shape[-1]
    if lengths is None:
        lengths = torch.full((len(target),), time, device=target.device, dtype=torch.long)
    mask = (torch.arange(time, device=target.device)[None] < lengths[:, None]).float()
    error = ((estimate[:, :, None] - target[:, None]).square() * mask[:, None, None]).sum(-1)
    energy = (target.square() * mask[:, None]).sum(-1)
    scale_error = 10 * torch.log10(1 + error / energy[:, None].clamp_min(1e-8))
    count = target.shape[1]
    orders = list(permutations(range(count)))
    si_candidates = torch.stack([scores[:, range(count), order].mean(-1) for order in orders], -1)
    scale_candidates = torch.stack([
        scale_error[:, range(count), order].mean(-1) for order in orders
    ], -1)
    joint = -si_candidates + scale_weight * scale_candidates
    indices = joint.argmin(-1, keepdim=True)
    metrics = {"loss": joint.gather(1, indices).mean(),
               "si_sdr": si_candidates.gather(1, indices).mean(),
               "scale_loss": scale_candidates.gather(1, indices).mean()}
    if return_assignment:
        metrics["assignment"] = torch.tensor(orders, device=target.device)[indices.squeeze(1)]
    return metrics


def pit_loss(estimate, target, lengths=None, scale_weight=0.0):
    if scale_weight == 0:
        return -pit_si_sdr(estimate, target, lengths)[0].mean()
    return pit_objective(estimate, target, lengths, scale_weight)["loss"]
