"""Shared greedy validation and paired evaluation for the particle games."""

import numpy as np
import torch
from torch.nn import functional as F

from fastap_agents.batched import select_actions


@torch.inference_mode()
def validation_returns(actor, spec, world):
    world.reset()
    mask = world.avail_actions()
    previous = torch.zeros_like(mask, dtype=torch.float32)
    returns = torch.zeros(world.batch_size, device=world.device)
    hidden = None
    for _ in range(world.episode_limit):
        actions, hidden = select_actions(actor, spec, world.observations(), mask, hidden, previous)
        reward, _, _ = world.step(actions)
        previous = F.one_hot(actions, spec.n_actions).float()
        returns += reward
    return returns.cpu().numpy()


def paired_return_difference(learned, control, seed=49711, resamples=10000):
    learned, control = np.asarray(learned, dtype=float), np.asarray(control, dtype=float)
    if learned.ndim != 1 or learned.shape != control.shape or not learned.size or resamples < 1:
        raise ValueError("Provide equal-length paired episode returns and a positive resample count")
    difference = learned - control
    rng = np.random.default_rng(seed)
    # Resample maps as pairs to retain the dependence on initial positions.
    estimates = []
    for start in range(0, resamples, 250):
        indices = rng.integers(difference.size, size=(min(250, resamples-start), difference.size))
        estimates.extend(difference[indices].mean(axis=1))
    return dict(mean_difference=float(difference.mean()),
                bootstrap_95_ci=np.quantile(estimates, [.025, .975]).tolist())
