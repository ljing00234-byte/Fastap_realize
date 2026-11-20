"""Paired evaluation of one QMIX team across three LBF start distributions."""

from math import comb
from operator import index

import numpy as np
import torch
from torch.nn import functional as F


STAGES = ("adjacent", "approach", "random")
_SPLIT_SEEDS = {"validation": 100_000, "heldout": 200_000}


def _summarize_pairs(trained, replacement, stage):
    """Rows are paired maps, columns are first-step, any food, all food, return."""
    names = ("first_step_success_rate", "success_rate", "full_completion_rate", "mean_return")
    difference = trained - replacement
    primary = 0 if stage == "adjacent" else 1
    wins = int((difference[:, primary] > 0).sum())
    losses = int((difference[:, primary] < 0).sum())
    discordant = wins + losses
    # One-sided exact paired test, conditional on maps where the outcomes differ.
    probability = sum(comb(discordant, k) for k in range(wins, discordant + 1)) / (2 ** discordant)
    trained_mean = trained.mean(axis=0)
    replacement_mean = replacement.mean(axis=0)
    passed = probability < 0.05 and (
        trained_mean[0] >= 0.80 if stage == "adjacent" else trained_mean[3] > replacement_mean[3]
    )
    return {
        "n_maps": len(trained),
        "trained": dict(zip(names, trained_mean.tolist())),
        "random_teammate": dict(zip(names, replacement_mean.tolist())),
        "paired_difference": dict(zip(names, difference.mean(axis=0).tolist())),
        "primary_metric": names[primary],
        "trained_only_successes": wins,
        "random_only_successes": losses,
        "paired_one_sided_p": probability,
        "criterion": ("first-step success >= 80% and paired P < 0.05" if stage == "adjacent"
                      else "paired P < 0.05 for any food and higher mean return"),
        "gate_passed": bool(passed),
    }


@torch.inference_mode()
def evaluate_cooperation(actor, spec, stage, *, split="validation", split_seed=None,
                         n_maps=128, device="cuda:0"):
    """Evaluate frozen slots 0/1 with either the trained or a random legal slot 2.

    A split reserves six consecutive seeds starting at split_seed, two per
    stage for the maps and replacement role. Use validation for training
    decisions and read heldout once for the final chosen team.
    """
    from fastap_methods.batched_qmix import select_actions
    from fastap_scenarios.torch_foraging import TorchForaging

    n_maps = index(n_maps)
    if stage not in STAGES or split not in _SPLIT_SEEDS or n_maps < 1:
        raise ValueError("Expected a known stage and split, and a positive map count")
    if (spec.scene_key, spec.control, spec.n_agents, spec.n_actions) != ("lbf", "team", 3, 6):
        raise ValueError("Cooperation validation requires an all-team, three-player LBF policy")

    split_seed = _SPLIT_SEEDS[split] if split_seed is None else index(split_seed)
    if not 0 <= split_seed <= 2 ** 64 - 2 * len(STAGES):
        raise ValueError("split_seed must leave room for all stage and replacement RNG seeds")
    map_seed = split_seed + 2 * STAGES.index(stage)
    action_seed = map_seed + 1
    initial = TorchForaging(n_maps, device=device, seed=map_seed)
    initial.reset(layout=stage)
    world = TorchForaging(2 * n_maps, device=device, seed=map_seed)
    world.load_snapshot(**{key: torch.cat((value, value), dim=0)
                           for key, value in initial.snapshot().items()})
    rng = torch.Generator(device=device).manual_seed(action_seed)
    previous = torch.zeros(2 * n_maps, spec.n_agents, spec.n_actions, device=device)
    returns = torch.zeros(2 * n_maps, device=device)
    completed = torch.zeros_like(returns, dtype=torch.bool)
    hidden = None
    training_mode = actor.training
    actor.eval()
    try:
        for t in range(world.episode_limit):
            available = world.avail_actions()
            actions, hidden = select_actions(actor, spec, world.observations(), available, hidden, previous)
            # An IID uniform score per action gives every legal action equal
            # probability of being chosen. Only role 2 of the second copy changes.
            legal = available[n_maps:, 2]
            scores = torch.rand(legal.shape, device=device, generator=rng)
            actions[n_maps:, 2] = scores.masked_fill(~legal.bool(), -1).argmax(dim=-1)
            previous = F.one_hot(actions, spec.n_actions).float()
            rewards, terminated, _ = world.step(actions)
            if t == 0:
                first = rewards > 0
            returns += rewards
            completed |= terminated
    finally:
        actor.train(training_mode)
    # Transfer at the end of the paired rollout. No per-map/device synchronization.
    outcomes = torch.stack((first, returns > 0, completed, returns), dim=1).cpu().numpy()
    trained, replacement = np.split(outcomes.astype(np.float64), 2)
    return {**_summarize_pairs(trained, replacement, stage), "stage": stage, "split": split,
            "split_seed": split_seed, "map_batch_seed": map_seed, "random_teammate_seed": action_seed}
