"""Shared greedy validation and paired evaluation for the particle games."""

import numpy as np
import torch
from torch.nn import functional as F

from fastap_agents.batched import select_actions


BOOTSTRAP_SEED = 49711
BOOTSTRAP_RESAMPLES = 10000


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


def paired_return_difference(learned, control, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
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


def serialize_scores(report):
    """Copy the individual episode returns only when publishing JSON evidence."""
    returns = report["episode_returns"]
    if not isinstance(returns, torch.Tensor):
        return report
    return {**report, "episode_returns": dict(zip(report["pool_order"], returns.tolist()))}


def compare_ego_returns(trained, initial, *, device=None):
    """Average over matched partners first, then bootstrap paired whole maps."""
    for key in ("first_map_seed", "n_maps_per_teammate", "pool_order", "board_generator", "board_device"):
        if trained.get(key) != initial.get(key):
            raise ValueError("Paired comparison requires the same starts and teammates")
    returns = [report["episode_returns"] for report in (trained, initial)]
    if device is None and all(isinstance(r, dict) for r in returns):
        if trained.get("scoring_device", "cpu") != "cpu":
            raise ValueError("Pair the original device returns before serializing CUDA scores")
        by_map = [np.mean([r[name] for name in trained["pool_order"]], axis=0) for r in returns]
        return paired_return_difference(*by_map)
    if not all(isinstance(r, torch.Tensor) for r in returns):
        raise ValueError("Paired device bootstrap requires both original return tensors")
    a, b = returns
    expected = torch.device(device) if device is not None else a.device
    if (a.device != b.device or a.device.type != expected.type
            or (expected.index is not None and a.device.index != expected.index)):
        raise ValueError("Paired returns must be on the requested evaluation device")
    shape = (len(trained["pool_order"]), trained["n_maps_per_teammate"])
    if a.shape != shape or b.shape != shape:
        raise ValueError("Unexpected dimensions for teammate-by-map returns")
    diff = (a - b).double().mean(dim=0)
    generator = torch.Generator(device=a.device).manual_seed(BOOTSTRAP_SEED)
    estimates = []
    for start in range(0, BOOTSTRAP_RESAMPLES, 250):
        count = min(250, BOOTSTRAP_RESAMPLES - start)
        indices = torch.randint(diff.numel(), (count, diff.numel()), device=a.device, generator=generator)
        estimates.append(diff[indices].mean(dim=1))
    limits = torch.quantile(torch.cat(estimates), diff.new_tensor([0.025, 0.975]))
    mean, low, high = torch.cat([diff.mean().reshape(1), limits]).tolist()
    return dict(mean_difference=mean, bootstrap_95_ci=[low, high], bootstrap_device=str(a.device),
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES)
