"""Stationary PP ego evaluation on the same seeded starts for every teammate."""

from operator import index

import numpy as np
import torch
from torch.nn import functional as F

from fastap_agents.batched import select_actions
from .particle_eval import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, compare_ego_returns, serialize_scores
from fastap_scenarios.predator_prey import PredatorPreyWorld
from fastap_scenarios.torch_pp_ego import TorchPPEgo


BOARD_GENERATOR = "torch_per_map_seed_uniform_float64_v1"


def make_evaluation_boards(first_seed, n_maps, *, device=None):
    """Each map seed identifies one start, independent of panel size and ordering."""
    first_seed, n_maps = index(first_seed), index(n_maps)
    if first_seed < 0 or n_maps < 1:
        raise ValueError("Expected a nonnegative first seed and at least one map")
    if device is not None:
        positions = torch.empty((n_maps, 6, 2), dtype=torch.float64, device=device)
        generator = torch.Generator(device=positions.device)
        for i in range(n_maps):
            generator.manual_seed(first_seed + i)
            torch.rand((6, 2), generator=generator, device=positions.device, out=positions[i])
        positions.mul_(2).sub_(1)
        positions[:, 4:].mul_(0.9)
        return dict(first_seed=first_seed, n_maps=n_maps, positions=positions,
                    velocity=torch.zeros_like(positions[:, :4]), board_generator=BOARD_GENERATOR,
                    board_device=str(positions.device))

    # Scalar reference starts are available for parity checks with the original PP game.
    positions, velocity = [], []
    for seed in range(first_seed, first_seed + n_maps):
        game = PredatorPreyWorld(seed=seed)
        try:
            game.reset()
            movers = [*game.players, game.prey]
            positions.append([body.position for body in (*movers, *game.obstacles)])
            velocity.append([body.velocity for body in movers])
        finally:
            game.close()
    return dict(first_seed=first_seed, n_maps=n_maps, positions=np.asarray(positions), velocity=np.asarray(velocity),
                board_generator="scalar_numpy_per_map_seed", board_device="cpu")


def _score_rows(returns, contacts):
    """Reduce the map dimension on the rollout device, retaining one row per group."""
    totals = contacts.sum(dim=2)
    tagged = totals > 0
    multiple = (contacts > 0).sum(dim=2) >= 2
    return torch.stack([returns.mean(1), tagged.sum(1), tagged.double().mean(1), totals.double().mean(1),
                        *contacts.double().mean(1).unbind(1), multiple.sum(1), multiple.double().mean(1)], dim=1)


def _score_record(row, n_episodes):
    mean_return, any_tag, any_rate, mean_contacts, ego0, ego1, teammate, multiple, multiple_rate = row
    return dict(n_episodes=n_episodes, mean_return=mean_return, any_tag=int(any_tag), any_tag_rate=any_rate,
                mean_contacts=mean_contacts, mean_contacts_by_role=[ego0, ego1, teammate],
                teammate_mean_contacts=teammate, multiple_predators_tagged=int(multiple),
                multiple_predators_tagged_rate=multiple_rate)


@torch.inference_mode()
def evaluate_pool_ego(actor, spec, teammates, boards, *, return_device_tensors=False):
    if ((spec.scene_key, spec.control, spec.n_agents, spec.obs_shape, spec.n_actions)
            != ("pp", "ego", 2, TorchPPEgo.obs_shape, TorchPPEgo.n_actions)) or not teammates:
        raise ValueError("PP evaluation needs a two-hunter ego policy and a nonempty frozen teammate pool")
    device = next(actor.parameters()).device
    n_maps = boards["n_maps"]
    if boards["board_device"] != str(device):
        raise ValueError("PP maps and policies must use the same evaluation device")
    if device.type == "cuda" and (boards["board_generator"] != BOARD_GENERATOR or
            any(not isinstance(boards[key], torch.Tensor) or boards[key].device != device
                for key in ("positions", "velocity"))):
        raise ValueError("CUDA PP evaluation requires maps generated on the same CUDA device")
    snapshot = {key: torch.as_tensor(boards[key], device=device) for key in ("positions", "velocity")}
    contacts, returns = [], []
    training = actor.training
    actor.eval()
    try:
        for teammate in teammates.values():
            world = TorchPPEgo(n_maps, device, teammates=[teammate])
            world.load_snapshot(**snapshot)
            mask = world.avail_actions()
            previous = torch.zeros_like(mask, dtype=torch.float32)
            hidden = None
            reward_sum = torch.zeros(n_maps, device=device)
            tag_counts = torch.zeros(n_maps, 3, dtype=torch.long, device=device)
            for _ in range(world.episode_limit):
                actions, hidden = select_actions(actor, spec, world.observations(), mask, hidden, previous)
                reward, _, _ = world.step(actions)
                reward_sum += reward
                tag_counts += world.game.last_contacts
                previous = F.one_hot(actions, spec.n_actions).float()
            contacts.append(tag_counts)
            returns.append(reward_sum)
    finally:
        actor.train(training)
    contacts = torch.stack(contacts)
    returns = torch.stack(returns).double()
    if not torch.equal(returns, 10 * contacts.sum(dim=2)):
        raise ValueError("PP team rewards disagree with counts of predator–prey contacts")
    per_member = _score_rows(returns, contacts)
    overall = _score_rows(returns.reshape(1, -1), contacts.reshape(1, -1, 3))
    records = torch.cat((overall, per_member)).tolist()
    report = dict(first_map_seed=boards["first_seed"], n_maps_per_teammate=n_maps, pool_order=list(teammates),
                  board_generator=boards["board_generator"], board_device=boards["board_device"],
                  policy_device=str(device), scoring_device=str(returns.device),
                  per_teammate={name: _score_record(row, n_maps) for name, row in zip(teammates, records[1:])},
                  episode_returns=returns, **_score_record(records[0], returns.numel()))
    return report if return_device_tensors else serialize_scores(report)
