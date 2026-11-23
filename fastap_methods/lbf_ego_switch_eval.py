"""Paired LBF ego evaluation with the within-episode roster draws of Fastap Algorithm 3."""

from operator import index

import torch
from torch.nn import functional as F

from fastap_agents.batched import select_actions
from fastap_methods.lbf_pool_ego_eval import ROSTERS
from fastap_methods.particle_eval import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from fastap_scenarios.foraging import _FOOD_LAYOUTS
from fastap_scenarios.torch_foraging import TorchForaging
from fastap_scenarios.torch_lbf_switch import TorchSwitchingLBFEgo


METRICS = ("mean_return", "any_food_rate", "all_food_rate")
EVENTS = ("redraws", "group_changes", "same_group_redraws", "one_to_two", "two_to_one",
          "episodes_with_group_change")
HORIZON = TorchForaging.episode_limit


def _map_noise(first_seed, n_maps, shape, device, *, stream=None):
    """Key draws by map so splitting the GPU panel never changes a map or its draw schedule."""
    values = torch.empty((n_maps, *shape), dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    for row, output in enumerate(values):
        map_seed = first_seed + row
        key = map_seed if stream is None else ((stream + 1) * 1_000_000_007 + map_seed) % (2**63)
        generator.manual_seed(key)
        torch.rand(output.shape, generator=generator, device=device, out=output)
    return values


def _boards(first_seed, n_maps, device):
    # The scalar game draws four player levels even if the fourth player initially sits out.
    # Keep the latent level, starting positions and food layout paired across initial roster sizes.
    noise = _map_noise(first_seed, n_maps, (41,), device)
    layouts = torch.tensor(_FOOD_LAYOUTS, device=device)
    foods = layouts[(noise[:, 0] * len(layouts)).long()]
    levels = 1 + (noise[:, 1:5] * 3).long()
    food_cells = foods[:, :, 0] * 6 + foods[:, :, 1]
    ranks = noise[:, 5:].scatter(1, food_cells, 2.)
    player_cells = ranks.topk(4, largest=False).indices
    positions = torch.stack((player_cells // 6, player_cells % 6), dim=-1)
    starts = []
    for size in (3, 4):
        food_level = levels[:, :size].sort(1).values[:, :3].sum(1, keepdim=True)
        field = torch.zeros((n_maps, 36), dtype=torch.long, device=device)
        field.scatter_(1, food_cells, food_level.expand(-1, 3))
        starts.append(dict(field=field.view(-1, 6, 6), positions=positions[:, :size], levels=levels[:, :size]))
    return starts, levels[:, 3]


def _draws(noise, n_policies, interval):
    """Event times count completed actions. Each event independently draws an ordered group."""
    lo, hi = interval
    times = (lo + (noise[..., 0] * (hi - lo + 1)).long()).cumsum(-1)
    first = (noise[..., 1] * n_policies).long()
    other = (noise[..., 2] * (n_policies - 1)).long()
    other = other + (other >= first)
    second = torch.where(noise[..., 3] < .5, -1, other)
    return times, torch.stack((first, second), dim=-1), noise[..., 4:]


def _initial_ids(maps, member, size, n_policies):
    other = (member + 1 + maps % (n_policies - 1)) % n_policies
    return torch.stack((member, other if size == 2 else torch.full_like(member, -1)), dim=-1)


def _rollout(world, ego, snapshot, fourth_levels, ids, draws):
    world.load_snapshot(**snapshot, fourth_levels=fourth_levels, teammate_ids=ids)
    previous = torch.zeros_like(world.avail_actions(), dtype=torch.float32)
    returns = previous.new_zeros(len(ids))
    counts = torch.zeros(len(ids), 5, device=ids.device, dtype=torch.long)
    hidden = None
    if draws is not None:
        times, groups, ranks = draws
        pointer = torch.zeros(len(ids), device=ids.device, dtype=torch.long)
        rows = torch.arange(len(ids), device=ids.device)
    for t in range(HORIZON):
        if draws is not None and t > 0:
            event = pointer.clamp_max(times.shape[1] - 1)
            due = (pointer < times.shape[1]) & (times[rows, event] == t) & ~(world.terminated | world.truncated)
            requested = torch.where(due[:, None], groups[rows, event], ids)
            changed = world.apply_teammates(requested, spawn_ranks=ranks[rows, event])
            counts[:, 0] += due
            counts[:, 1] += changed
            counts[:, 2] += due & ~changed
            counts[:, 3] += due & (ids[:, 1] < 0) & (requested[:, 1] >= 0)
            counts[:, 4] += due & (ids[:, 1] >= 0) & (requested[:, 1] < 0)
            pointer += due
            ids = requested
        actions, hidden = select_actions(ego.model, ego.spec, world.observations(), world.avail_actions(),
                                         hidden, previous)
        reward, _, _ = world.step(actions)
        returns += reward
        previous = F.one_hot(actions, ego.spec.n_actions).float()
    had_group_change = counts[:, 1:2] > 0
    counts = torch.cat((counts, had_group_change), dim=-1)
    return returns, counts


def _panel_summary(values):
    return dict(zip(METRICS, values.mean(dim=(0, 1, 2)).tolist()))


def _paired(values, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    """Resample complete map IDs; keep both conditions, all initial partners and both sizes paired."""
    by_roster_and_map = values.mean(dim=2)
    by_map = torch.cat((by_roster_and_map.mean(1, keepdim=True), by_roster_and_map), dim=1)
    generator = torch.Generator(device=values.device).manual_seed(seed)
    estimates = []
    for start in range(0, resamples, 100):
        indices = torch.randint(len(by_map), (min(100, resamples-start), len(by_map)),
                                generator=generator, device=values.device)
        estimates.append(by_map[indices].mean(1))
    intervals = torch.quantile(torch.cat(estimates), values.new_tensor([.025, .975]), dim=0).tolist()
    means = by_map.mean(0).tolist()
    out = []
    for r, row in enumerate(means):
        out.append({name: dict(mean_difference=v, bootstrap_95_ci=[intervals[0][r][m], intervals[1][r][m]])
                    for m, (name, v) in enumerate(zip(METRICS, row))})
    return dict(**out[0], per_initial_roster=dict(zip(ROSTERS, out[1:])), bootstrap_seed=seed,
                bootstrap_resamples=resamples, unit="whole paired map", direction="switching minus stationary")


@torch.inference_mode()
def evaluate_ego_switch(ego, teammates, *, first_seed, n_maps, schedule_seed, interval=(5, 8), chunk_size=256):
    """Greedy fixed-versus-redrawn teams on shared GPU starts; return a JSON-ready paired report."""
    first_seed, schedule_seed, n_maps, chunk_size = map(index, (first_seed, schedule_seed, n_maps, chunk_size))
    lo, hi = map(index, interval)
    members = list(teammates.values())
    n_policies = len(members)
    if (min(first_seed, schedule_seed) < 0 or min(n_maps, chunk_size) < 1 or not 1 <= lo <= hi
            or n_policies < 2 or (ego.spec.scene_key, ego.spec.control, ego.spec.n_agents) != ("lbf", "ego", 2)):
        raise ValueError("Expected an LBF ego, at least two teammates, positive panel sizes and waiting times")
    device = next(ego.model.parameters()).device
    board_starts, fourth_levels = _boards(first_seed, n_maps, device)
    n_events = max(1, (HORIZON - 1) // lo)
    noise = _map_noise(first_seed, n_maps, (2, n_policies, n_events, 40), device, stream=schedule_seed)
    returns = torch.empty(2, n_maps, 2, n_policies, device=device, dtype=torch.float64)
    event_counts = torch.zeros(n_maps, 2, n_policies, len(EVENTS), dtype=torch.long, device=device)
    training = ego.model.training
    ego.model.eval()
    try:
        for roster, snapshot in enumerate(board_starts):
            for start in range(0, n_maps * n_policies, chunk_size):
                flat = torch.arange(start, min(start + chunk_size, n_maps * n_policies), device=device)
                maps, primary = flat // n_policies, flat % n_policies
                ids = _initial_ids(maps, primary, roster + 1, n_policies)
                boards = {k: v[maps] for k, v in snapshot.items()}
                draws = _draws(noise[maps, roster, primary], n_policies, (lo, hi))
                world = TorchSwitchingLBFEgo(len(flat), device, teammates=members)
                for mode, schedule in enumerate((None, draws)):
                    scores, counts = _rollout(world, ego, boards, fourth_levels[maps], ids, schedule)
                    returns[mode, maps, roster, primary] = scores.double()
                    if mode == 1:
                        event_counts[maps, roster, primary] = counts
    finally:
        ego.model.train(training)

    # All three foods initially have equal value and remain fixed throughout a game.
    outcomes = torch.stack((returns, (returns > 1/6).double(), (returns > 5/6).double()), dim=-1)
    if (event_counts[..., 0] != event_counts[..., 1] + event_counts[..., 2]).any():
        raise RuntimeError("Redraw accounting disagrees with actual roster changes")
    results = {}
    for mode, label in enumerate(("stationary", "switching")):
        vals = outcomes[mode]
        results[label] = dict(n_episodes=n_maps * 2 * n_policies, **_panel_summary(vals),
                              per_initial_roster={name: dict(n_episodes=n_maps*n_policies,
                                                             **_panel_summary(vals[:, i:i+1]))
                                                  for i, name in enumerate(ROSTERS)})
    events = dict(zip(EVENTS, event_counts.sum(dim=(0, 1, 2)).tolist()))
    events["size_changes"] = events["one_to_two"] + events["two_to_one"]
    return dict(first_seed=first_seed, n_maps=n_maps, schedule_seed=schedule_seed, interval=[lo, hi],
                pool_order=list(teammates), device=str(device), horizon=HORIZON,
                board_generator="torch_map_keyed_lbf_layout_levels_free_positions_v1",
                paired=_paired(outcomes[1] - outcomes[0]), events=events, **results)
