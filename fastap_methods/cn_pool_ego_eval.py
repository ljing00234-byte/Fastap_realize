"""Evaluate two navigation egos with each frozen teammate on identical maps."""

from operator import index

import torch
from torch.nn import functional as F

from fastap_agents.batched import select_actions
from fastap_scenarios.cooperative_navigation import DEFAULT_COLLISION_PENALTY
from fastap_scenarios.torch_cn_ego import TorchCNEgo
from .particle_eval import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, compare_ego_returns, serialize_scores


BOARD_GENERATOR = "torch_cn_per_map_seed_uniform_float64_v1"


def make_evaluation_boards(first_seed, n_maps, *, device="cpu"):
    """Use CN's random reset distribution, with a separate seed for each board."""
    first_seed, n_maps = index(first_seed), index(n_maps)
    if first_seed < 0 or n_maps < 1:
        raise ValueError("Expected a nonnegative first seed and at least one CN map")
    coords = torch.empty(n_maps, 7, 2, dtype=torch.float64, device=device)
    generator = torch.Generator(device=coords.device)
    for offset, board in enumerate(coords):
        generator.manual_seed(first_seed + offset)
        torch.rand(board.shape, generator=generator, device=coords.device, out=board)
    coords.mul_(2).sub_(1)
    return dict(first_seed=first_seed, n_maps=n_maps, positions=coords[:, 4:], landmarks=coords[:, :4],
                velocity=torch.zeros_like(coords[:, 4:]), board_generator=BOARD_GENERATOR,
                board_device=str(coords.device))


@torch.inference_mode()
def evaluate_pool_ego(actor, spec, teammates, boards, *, return_device_tensors=False):
    if ((spec.scene_key, spec.control, spec.n_agents, spec.obs_shape, spec.n_actions)
            != ("cn", "ego", 2, TorchCNEgo.obs_shape, TorchCNEgo.n_actions)) or not teammates:
        raise ValueError("CN evaluation needs two ego agents and a nonempty frozen teammate pool")
    device = next(actor.parameters()).device
    snapshot = {key: boards[key] for key in ("positions", "velocity", "landmarks")}
    if (boards["board_device"] != str(device) or boards["board_generator"] != BOARD_GENERATOR
            or any(not isinstance(value, torch.Tensor) or value.device != device for value in snapshot.values())):
        raise ValueError("Expected seeded CN maps on the evaluation device")
    n_maps = boards["n_maps"]
    scores = []
    training = actor.training
    actor.eval()
    try:
        for teammate in teammates.values():
            world = TorchCNEgo(n_maps, device, teammates=[teammate])
            world.load_snapshot(**snapshot)
            mask = world.avail_actions()
            previous = torch.zeros_like(mask, dtype=torch.float32)
            hidden = None
            totals = torch.zeros(3, n_maps, dtype=torch.float64, device=device)
            returns, distances, collisions = totals.unbind()
            for _ in range(world.episode_limit):
                actions, hidden = select_actions(actor, spec, world.observations(), mask, hidden, previous)
                reward, _, _ = world.step(actions)
                returns.add_(reward)
                distances.add_(world.game.distance_sum)
                collisions.add_(world.game.collisions)
                previous = F.one_hot(actions, spec.n_actions).float()
            scores.append(totals)
    finally:
        actor.train(training)

    # Keep the raw environment return and the two independently measured reward components.
    returns, distances, collisions = torch.stack(scores).unbind(1)
    if not torch.allclose(returns, -distances - DEFAULT_COLLISION_PENALTY * collisions, rtol=1e-6, atol=1e-6):
        raise ValueError("CN team rewards disagree with landmark distances and pair collisions")
    records = torch.stack((returns.mean(1), distances.mean(1) / TorchCNEgo.episode_limit,
                           collisions.mean(1)), dim=1)
    aggregate, *by_teammate = torch.cat((records.mean(0, keepdim=True), records)).tolist()
    fields = ("mean_return", "mean_landmark_distance_sum_per_step", "mean_pair_collisions_per_episode")
    report = dict(first_map_seed=boards["first_seed"], n_maps_per_teammate=n_maps,
                  n_episodes=returns.numel(), pool_order=list(teammates),
                  board_generator=BOARD_GENERATOR, board_device=boards["board_device"],
                  policy_device=str(device), scoring_device=str(returns.device), episode_returns=returns,
                  per_teammate={name: dict(n_episodes=n_maps, **dict(zip(fields, row)))
                                for name, row in zip(teammates, by_teammate)}, **dict(zip(fields, aggregate)))
    return report if return_device_tensors else serialize_scores(report)
