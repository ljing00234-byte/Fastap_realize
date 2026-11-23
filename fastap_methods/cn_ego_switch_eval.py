"""Evaluate CN ego policies with changing teammate identities and roster size."""

from operator import index

import torch
from torch.nn import functional as F

from fastap_agents.batched import select_actions
from fastap_methods.cn_pool_ego_eval import BOARD_GENERATOR, evaluate_pool_ego
from fastap_methods.particle_eval import compare_ego_returns
from fastap_scenarios.cooperative_navigation import DEFAULT_COLLISION_PENALTY
from fastap_scenarios.torch_cn_switch import TorchSwitchingCNEgo


def make_switch_schedule(boards, pool_order, *, schedule_seed, interval=(5, 8)):
    """Key event times, new groups and arrival positions by map and starting teammate."""
    lo, hi = map(index, interval)
    seed = index(schedule_seed)
    names = list(pool_order)
    if seed < 0 or not 1 <= lo <= hi or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Expected distinct teammates, a nonnegative seed and positive waiting times")
    if boards["board_generator"] != BOARD_GENERATOR or boards["n_maps"] < 1:
        raise ValueError("Use seeded CN maps")
    device = boards["positions"].device
    n_members, n_maps, horizon = len(names), boards["n_maps"], TorchSwitchingCNEgo.episode_limit
    n_events = max(1, (horizon - 1) // lo)
    noise = torch.empty((n_maps, n_members, n_events, 6), dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    for offset, row in enumerate(noise):
        map_seed = boards["first_seed"] + offset
        generator.manual_seed(((seed + 1) * 1_000_000_007 + map_seed) % 2**63)
        torch.rand(row.shape, generator=generator, device=device, out=row)
    noise = noise.transpose(0, 1)
    times = (lo + (noise[..., 0] * (hi - lo + 1)).long()).cumsum(dim=-1)
    first = (noise[..., 1] * n_members).long()
    other = (noise[..., 2] * (n_members - 1)).long()
    second = torch.where(noise[..., 3] < 0.5, -1, other + (other >= first))
    groups = torch.stack((first, second), dim=-1)
    initial = torch.arange(n_members, device=device)[:, None, None].expand(-1, n_maps, 1)
    initial = torch.stack((initial, torch.full_like(initial, -1)), dim=-1)
    groups = torch.cat((initial, groups), dim=2)
    spawns = F.pad(2 * noise[..., 4:] - 1, (0, 0, 1, 0))
    decisions = torch.arange(horizon, device=device).expand(n_members, n_maps, -1).contiguous()
    event = torch.searchsorted(times.contiguous(), decisions, right=True)
    due = event.diff(dim=-1, prepend=event[..., :1]) > 0
    where = event.unsqueeze(-1).expand(-1, -1, -1, 2)
    return groups.gather(2, where), spawns.gather(2, where), due


def _evaluate_redraws(ego, teammates, boards, schedule, *, allow_arrivals):
    ids, spawn, due = schedule
    if not allow_arrivals:
        ids = ids.clone()
        ids[..., 1] = -1
    members = list(teammates.values())
    n_maps, horizon = boards["n_maps"], TorchSwitchingCNEgo.episode_limit
    snapshot = {key: boards[key] for key in ("positions", "velocity", "landmarks")}
    metrics = torch.zeros((len(members), 3, n_maps), dtype=torch.float64, device=ids.device)
    changes = torch.zeros_like(due)
    for i in range(len(members)):
        world = TorchSwitchingCNEgo(n_maps, ids.device, teammates=members)
        world.load_snapshot(**snapshot, teammate_ids=ids[i, :, 0])
        mask = world.avail_actions()
        previous = torch.zeros_like(mask, dtype=torch.float32)
        hidden = None
        returns, distances, collisions = metrics[i].unbind()
        for t in range(horizon):
            if t:
                changes[i, :, t] = world.apply_teammates(ids[i, :, t], spawn_positions=spawn[i, :, t])
            actions, hidden = select_actions(ego.model, ego.spec, world.observations(), mask, hidden, previous)
            reward, _, _ = world.step(actions)
            returns.add_(reward)
            distances.add_(world.game.distance_sum)
            collisions.add_(world.game.collisions)
            previous = F.one_hot(actions, ego.spec.n_actions).float()
    returns, distances, collisions = metrics.unbind(1)
    if not torch.allclose(returns, -distances - DEFAULT_COLLISION_PENALTY * collisions, rtol=1e-6, atol=1e-6):
        raise RuntimeError("CN rewards disagree with the landmark distances and pair collisions")
    if (changes & ~due).any():
        raise RuntimeError("CN group changed between scheduled redraws")
    fields = ("mean_return", "mean_landmark_distance_sum_per_step", "mean_pair_collisions_per_episode")
    summaries = torch.stack((returns.mean(1), distances.mean(1) / horizon, collisions.mean(1)), dim=1)
    overall, *rows = torch.cat((summaries.mean(0, keepdim=True), summaries)).tolist()
    report = dict(first_map_seed=boards["first_seed"], n_maps_per_teammate=n_maps, n_episodes=returns.numel(),
                  pool_order=list(teammates), board_generator=BOARD_GENERATOR, board_device=boards["board_device"],
                  policy_device=str(ids.device), scoring_device=str(ids.device), episode_returns=returns,
                  per_teammate={name: dict(n_episodes=n_maps, **dict(zip(fields, row)))
                                for name, row in zip(teammates, rows)}, **dict(zip(fields, overall)))
    present = ids[..., 1] >= 0
    size_change = present.long().diff(dim=-1)
    event_fields = ("redraws", "group_changes", "same_group_redraws", "one_to_two", "two_to_one",
                    "episodes_with_group_change", "two_teammate_steps")
    counts = (due.sum(), changes.sum(), (due & ~changes).sum(), (size_change > 0).sum(),
              (size_change < 0).sum(), changes.any(-1).sum(), present.sum())
    return report, dict(zip(event_fields, torch.stack(counts).tolist()))


@torch.inference_mode()
def evaluate_ego_switch(ego, teammates, boards, *, schedule_seed, interval=(5, 8), stationary=None):
    """Pair the two redraw conditions with each other and with matching stationary held-out scores."""
    device = next(ego.model.parameters()).device
    if ((ego.spec.scene_key, ego.spec.control, ego.spec.n_agents) != ("cn", "ego", 2)
            or any(boards[k].device != device for k in ("positions", "velocity", "landmarks"))
            or boards["board_device"] != str(device)):
        raise ValueError("Expected CN maps and both ego roles on the evaluation device")
    if stationary is None:
        stationary = evaluate_pool_ego(ego.model, ego.spec, teammates, boards, return_device_tensors=True)
    if (stationary["first_map_seed"] != boards["first_seed"] or stationary["pool_order"] != list(teammates)
            or stationary["n_maps_per_teammate"] != boards["n_maps"] or stationary["board_generator"] != BOARD_GENERATOR
            or stationary["board_device"] != str(device) or stationary["episode_returns"].device != device
            or stationary["episode_returns"].shape != (len(teammates), boards["n_maps"])):
        raise ValueError("Stationary baseline must match the CN maps, pool order and scoring device")
    schedule = make_switch_schedule(boards, teammates, schedule_seed=schedule_seed, interval=interval)
    training = ego.model.training
    ego.model.eval()
    try:
        policy, policy_events = _evaluate_redraws(ego, teammates, boards, schedule, allow_arrivals=False)
        opened, open_events = _evaluate_redraws(ego, teammates, boards, schedule, allow_arrivals=True)
    finally:
        ego.model.train(training)
    return dict(stationary=stationary, policy_switch=policy, open_switch=opened,
                paired=dict(policy_minus_stationary=compare_ego_returns(policy, stationary, device=device),
                            open_minus_stationary=compare_ego_returns(opened, stationary, device=device),
                            open_minus_policy=compare_ego_returns(opened, policy, device=device)),
                events=dict(policy_switch=policy_events, open_switch=open_events))
