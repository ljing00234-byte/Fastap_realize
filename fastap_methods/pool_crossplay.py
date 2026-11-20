"""Evaluate all original-partner/teammate combinations on identical, fresh LBF boards."""

import numpy as np
import torch
from torch.nn import functional as F

from fastap_agents import load_frozen_agent
from fastap_agents.batched import select_actions
from fastap_scenarios.foraging import ForagingWorld
from fastap_scenarios.torch_foraging import TorchForaging


def _initial_boards(first_seed, count):
    fields, positions, levels = [], [], []
    for seed in range(first_seed, first_seed + count):
        board = ForagingWorld(seed=seed)
        try:
            board.reset(1)
            fields.append(board.field.copy())
            positions.append([p.position for p in board.players])
            levels.append([p.level for p in board.players])
        finally:
            board.close()
    return dict(field=np.stack(fields), positions=positions, levels=levels)


def evaluate_pool(directory, members, first_seed, n_maps):
    """Rows supply slots 0 and 1. Columns supply slot 2; policies are frozen and greedy."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _evaluate(directory, members, first_seed, n_maps)
    finally:
        torch.set_num_threads(previous_threads)


@torch.inference_mode()
def _evaluate(directory, members, first_seed, n_maps):
    agents = [load_frozen_agent(directory / m["path"], device="cpu") for m in members]
    if not agents or any((a.spec.scene_key, a.spec.n_agents, a.spec.control) != ("lbf", 3, "team")
                         for a in agents):
        raise ValueError("Expected frozen three-player LBF teams")
    maps = _initial_boards(first_seed, n_maps)
    world = TorchForaging(n_maps, "cpu", seed=first_seed)
    matrix = []
    reference = []
    for i, ego in enumerate(agents):
        row = []
        for j, teammate in enumerate(agents):
            world.load_snapshot(**maps)
            previous = torch.zeros(n_maps, world.n_agents, world.n_actions)
            ego_hidden = mate_hidden = None
            returns = torch.zeros(n_maps)
            for _ in range(world.episode_limit):
                obs, available = world.observations(), world.avail_actions()
                actions, ego_hidden = select_actions(ego.model, ego.spec, obs, available, ego_hidden, previous)
                if i != j:
                    other, mate_hidden = select_actions(teammate.model, teammate.spec, obs, available,
                                                        mate_hidden, previous)
                    actions[:, 2] = other[:, 2]
                if i == 0 and j == 0:
                    reference.append((obs, available, previous, ~(world.terminated | world.truncated)))
                reward, _, _ = world.step(actions)
                returns += reward
                previous = F.one_hot(actions, world.n_actions).float()
            row.append(dict(mean_return=returns.mean().item(), any_food=int((returns > 1/6).sum()),
                            all_food=int(world.terminated.sum())))
        matrix.append(row)
    # Compare actions on the same local history, including role 2's actual previous action.
    predictions = []
    mask = torch.stack([active for _, _, _, active in reference])
    for agent in agents:
        choices = []
        hidden = None
        for obs, available, previous, _ in reference:
            actions, hidden = select_actions(agent.model, agent.spec, obs, available, hidden, previous)
            choices.append(actions[:, 2])
        predictions.append(torch.stack(choices)[mask])
    disagreement = [[(a != b).float().mean().item() for b in predictions] for a in predictions]
    return dict(ids=[m["id"] for m in members], n_maps=n_maps, first_map_seed=first_seed,
                rows="original trained partners in slots 0/1", columns="evaluated frozen teammate in slot 2",
                crossplay=matrix, reference_team=members[0]["id"],
                reference_decisions=int(mask.sum()), action_disagreement=disagreement)
