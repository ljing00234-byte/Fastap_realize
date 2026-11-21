"""Stationary evaluation of two LBF egos on balanced three- and four-player teams."""

from operator import index

import numpy as np
import torch
from torch.nn import functional as F

from fastap_agents.batched import select_actions
from fastap_methods.particle_eval import paired_return_difference
from fastap_methods.pool_crossplay import _initial_boards
from fastap_scenarios.torch_lbf_ego import TorchLBFEgo


ROSTERS = ("one_teammate", "two_teammates")


def make_evaluation_boards(first_seed, n_maps):
    """Build each roster natively using the scalar world with the same map seeds."""
    first_seed, n_maps = index(first_seed), index(n_maps)
    if first_seed < 0 or n_maps < 1:
        raise ValueError("Expected a nonnegative first seed and at least one map")
    snapshots = {label: _initial_boards(first_seed, n_maps, size)
                 for size, label in enumerate(ROSTERS, start=1)}
    return dict(first_seed=first_seed, n_maps=n_maps, snapshots=snapshots)


def _scores(returns):
    # Both roster sizes generate three equally valued foods at reset.
    return dict(mean_return=float(returns.mean()), any_food_rate=float((returns > 1/6).mean()),
                all_food_rate=float((returns > 5/6).mean()))


def _stationary_returns(actor, spec, world):
    previous = torch.zeros(world.batch_size, spec.n_agents, spec.n_actions, device=world.device)
    returns = torch.zeros(world.batch_size, device=world.device)
    hidden = None
    for _ in range(world.episode_limit):
        actions, hidden = select_actions(actor, spec, world.observations(), world.avail_actions(),
                                         hidden, previous)
        reward, _, _ = world.step(actions)
        returns += reward
        previous = F.one_hot(actions, spec.n_actions).float()
    return returns


@torch.inference_mode()
def evaluate_pool_ego(actor, spec, teammates, boards):
    """Hold the group fixed within each episode. Give each roster half the weight.

    Each map sees each pool member once in slot 2. For four players, slot 3
    cycles through the other members with a deterministic map-specific offset.
    Every pool member appears exactly once in each slot on each map; each
    ordered pair appears once in every complete block of (pool size - 1) maps.
    """
    if ((spec.scene_key, spec.control, spec.n_agents, spec.obs_shape, spec.n_actions)
            != ("lbf", "ego", 2, TorchLBFEgo.obs_shape, TorchLBFEgo.n_actions)):
        raise ValueError("Evaluation requires a two-role LBF ego policy")
    names = list(teammates)
    n_members = len(names)
    if n_members < 2:
        raise ValueError("Expected at least two distinct frozen teammate policies")
    n_maps = boards["n_maps"]
    device = next(actor.parameters()).device
    offsets = 1 + np.arange(n_maps) % (n_members - 1)
    per_roster, episode_returns, roster_arrays = {}, {}, []
    training = actor.training
    actor.eval()
    try:
        for size, label in enumerate(ROSTERS, start=1):
            snapshot = {k: torch.as_tensor(x, device=device) for k, x in boards["snapshots"][label].items()}
            world = TorchLBFEgo(n_maps, device, teammates=list(teammates.values()), n_teammates=size)
            member_returns = []
            for i in range(n_members):
                first = np.full(n_maps, i, dtype=np.int64)
                ids = first if size == 1 else np.stack((first, (i+offsets) % n_members), axis=1)
                world.load_snapshot(**snapshot, teammate_ids=torch.as_tensor(ids, device=device))
                member_returns.append(_stationary_returns(actor, spec, world))
            rows = torch.stack(member_returns).cpu().numpy().astype(np.float64)
            roster_arrays.append(rows)
            episode_returns[label] = {name: row.tolist() for name, row in zip(names, rows)}
            pairings = {}
            for i, (name, row) in enumerate(zip(names, rows)):
                if size == 1:
                    pairings[name] = dict(n_maps=n_maps, **_scores(row))
                else:
                    for off in np.unique(offsets):
                        partner = names[(i+off) % n_members]
                        subset = row[offsets == off]
                        pairings[f"{name}+{partner}"] = dict(n_maps=len(subset), **_scores(subset))
            per_roster[label] = dict(n_episodes=int(rows.size), per_pairing=pairings, **_scores(rows))
    finally:
        actor.train(training)
    return dict(first_map_seed=boards["first_seed"], n_maps_per_teammate=n_maps, n_episodes=2*n_members*n_maps,
                pool_order=names, pairing_rule="slot3 index = (slot2 index + 1 + map offset % (pool size - 1)) % pool size",
                per_roster=per_roster, episode_returns=episode_returns, **_scores(np.stack(roster_arrays)))


def compare_ego_returns(trained, initial):
    """Pair by seed and roster, then bootstrap whole maps with every pairing kept together."""
    for key in ("first_map_seed", "n_maps_per_teammate", "pool_order", "pairing_rule"):
        if trained[key] != initial[key]:
            raise ValueError("Paired comparison requires identical maps and pairing schedules")
    def map_means(result):
        return np.stack([np.mean([result["episode_returns"][label][name] for name in result["pool_order"]], axis=0)
                         for label in ROSTERS])
    a, b = map_means(trained), map_means(initial)
    overall = paired_return_difference(a.mean(axis=0), b.mean(axis=0))
    overall["per_roster"] = {label: paired_return_difference(a[i], b[i]) for i, label in enumerate(ROSTERS)}
    return overall
