"""Paired PP evaluation of fixed and changing policies for the third predator."""

from operator import index

import torch
from torch.nn import functional as F

from fastap_agents.batched import select_actions
from fastap_methods.particle_eval import compare_ego_returns
from fastap_methods.pp_pool_ego_eval import BOARD_GENERATOR, _score_record, _score_rows, evaluate_pool_ego
from fastap_scenarios.torch_pp_ego import TorchPPEgo


def make_switch_schedule(boards, pool_order, *, schedule_seed, interval=(5, 8)):
    """Draw one schedule per (map seed, starting teammate), shared by all tested ego policies."""
    lo, hi = map(index, interval)
    seed = index(schedule_seed)
    names = list(pool_order)
    if seed < 0 or not 1 <= lo <= hi or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Expected distinct teammates, a nonnegative schedule seed and positive waiting times")
    if boards["board_generator"] != BOARD_GENERATOR or boards["n_maps"] <= 0:
        raise ValueError("Use seeded PP device boards")
    device = boards["positions"].device
    n_maps, n_mates, horizon = boards["n_maps"], len(names), TorchPPEgo.episode_limit
    n_events = max(1, (horizon - 1) // lo)
    shape = (n_mates, n_maps, n_events)
    waits = torch.empty(shape, device=device, dtype=torch.long)
    draws = torch.empty_like(waits)
    rng = torch.Generator(device=device)
    for row in range(n_maps):
        map_seed = boards["first_seed"] + row
        rng.manual_seed(((seed + 1) * 1_000_000_007 + map_seed) % 2**63)
        waits[:, row] = torch.randint(lo, hi + 1, (n_mates, n_events), generator=rng, device=device)
        draws[:, row] = torch.randint(n_mates, (n_mates, n_events), generator=rng, device=device)
    times = waits.cumsum(-1)
    # Zero-based decision t uses the new policy when exactly t actions have elapsed.
    steps = torch.arange(horizon, device=device).expand(n_mates, n_maps, -1).contiguous()
    completed = torch.searchsorted(times, steps, right=True)
    initial = torch.arange(n_mates, device=device)[:, None, None].expand(-1, n_maps, 1)
    policies = torch.cat((initial, draws), dim=-1).gather(-1, completed)
    due = completed.diff(dim=-1, prepend=completed[..., :1]) > 0
    return dict(first_map_seed=boards["first_seed"], n_maps=n_maps, pool_order=names, schedule_seed=seed,
                interval=[lo, hi], policy_ids=policies, redraws=due)


@torch.inference_mode()
def evaluate_ego_switch(ego, teammates, boards, schedule):
    """Return both panels with device-resident episode returns indexed by starting partner and map."""
    names = list(teammates)
    if (names != schedule["pool_order"] or boards["first_seed"] != schedule["first_map_seed"]
            or boards["n_maps"] != schedule["n_maps"]):
        raise ValueError("Schedule must match the boards and ordered PP teammate pool")
    actor = ego.model
    static = evaluate_pool_ego(actor, ego.spec, teammates, boards, return_device_tensors=True)
    device = static["episode_returns"].device
    n_maps, horizon = boards["n_maps"], TorchPPEgo.episode_limit
    ids, due = schedule["policy_ids"], schedule["redraws"]
    if ids.device != device or due.device != device or ids.shape != (len(names), n_maps, horizon) or due.shape != ids.shape:
        raise ValueError("Expected one schedule on the evaluation device per initial partner and map")
    snapshot = {k: boards[k] for k in ("positions", "velocity")}
    contacts = torch.zeros(len(names), n_maps, 3, dtype=torch.long, device=device)
    returns = torch.zeros_like(static["episode_returns"])
    changes = torch.zeros_like(due)
    training = actor.training
    actor.eval()
    try:
        for i in range(len(names)):
            world = TorchPPEgo(n_maps, device, teammates=list(teammates.values()))
            world.load_snapshot(**snapshot, teammate_ids=ids[i, :, 0])
            previous = torch.zeros_like(world.avail_actions(), dtype=torch.float32)
            hidden = None
            for t in range(horizon):
                if t:
                    changes[i, :, t] = world.apply_teammates(ids[i, :, t])
                actions, hidden = select_actions(actor, ego.spec, world.observations(), world.avail_actions(),
                                                 hidden, previous)
                reward, _, _ = world.step(actions)
                returns[i] += reward
                contacts[i] += world.game.last_contacts
                previous = F.one_hot(actions, ego.spec.n_actions).float()
    finally:
        actor.train(training)

    if not torch.equal(returns, 10 * contacts.sum(-1)) or (changes & ~due).any():
        raise RuntimeError("PP contact rewards or policy changes disagree with the paired test protocol")
    overall = _score_rows(returns.reshape(1, -1), contacts.reshape(1, -1, 3))
    aggregate, *rows = torch.cat((overall, _score_rows(returns, contacts))).tolist()
    switched = {**static, **_score_record(aggregate, returns.numel()),
                "episode_returns": returns,
                "per_teammate": {name: _score_record(row, n_maps) for name, row in zip(names, rows)}}
    # Keep the old score format: per_teammate here labels the teammate at the start of the episode.
    paired = {"mean_return": compare_ego_returns(switched, static, device=device)}
    paired["any_tag_rate"] = compare_ego_returns({**switched, "episode_returns": (returns > 0).double()},
                                                  {**static, "episode_returns": (static["episode_returns"] > 0).double()},
                                                  device=device)
    n_due, n_changed, n_episodes_changed = torch.stack((due.sum(), changes.sum(), changes.any(-1).sum())).tolist()
    events = dict(redraws=n_due, group_changes=n_changed, same_group_redraws=n_due - n_changed,
                  episodes_with_group_change=n_episodes_changed)
    return dict(stationary=static, switching=switched, paired=paired, events=events)
