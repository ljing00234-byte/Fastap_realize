"""Cross-episode context experiment on the eight known probe streams."""

import numpy as np
import torch

from fastap_methods.fastap_crp import _collate


def fit_cross_episode(crp, probes, steps, peer_rng):
    """Fit on evidence from another episode of the target teammate. Return the verified pair count."""
    # The current group is fitted before _assign appends its slice.
    slices = [*crp.group_slices, slice(len(crp._probes), len(probes))]
    starts = np.empty(len(probes), dtype=np.int64)
    lengths = np.empty_like(starts)
    for group in slices:
        count = group.stop - group.start
        if count < 2:
            raise ValueError("Need at least two probe episodes per teammate")
        starts[group], lengths[group] = group.start, count
    pairs = 0
    crp.model.train()
    for _ in range(steps):
        target_ids = torch.randperm(len(probes), device='cpu')[:crp.minibatch_size].numpy()
        # Add a uniform nonzero offset modulo the group's episode count.
        offsets = peer_rng.integers(1, lengths[target_ids])
        first = starts[target_ids]
        peer_ids = first + (target_ids - first + offsets) % lengths[target_ids]
        if np.any(peer_ids == target_ids) or np.any(starts[peer_ids] != first):
            raise RuntimeError("Context and target must be different episodes of the same teammate")
        pairs += len(peer_ids)
        targets = _collate([probes[i] for i in target_ids], crp.device)
        peers = _collate([probes[i] for i in peer_ids], crp.device)
        context = crp.model.encode(*peers)
        loss = crp._model_loss(crp.model.trajectory_log_prob(*targets, context))
        crp.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(crp.model.parameters(), 10., error_if_nonfinite=True)
        crp.optimizer.step()
    crp.last_model_loss = loss.detach().item()
    return pairs
