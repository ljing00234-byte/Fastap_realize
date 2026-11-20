"""Batched PP with the scalar world's default evading prey and MPE dynamics."""

import numpy as np
import torch

from .particle import MOVES, _CONTACT_FORCE, _CONTACT_MARGIN, _DAMPING, _DT
from .predator_prey import PredatorPreyWorld


class TorchPredatorPrey:
    scene_key = "pp"
    n_agents = 3
    n_actions = PredatorPreyWorld.n_actions
    obs_shape = PredatorPreyWorld.obs_shape
    state_shape = PredatorPreyWorld.state_shape
    episode_limit = 25

    def __init__(self, batch_size, device, seed=0):
        if batch_size < 1:
            raise ValueError("PP requires at least one board")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.moves = torch.as_tensor(MOVES, device=device)
        self.acceleration = self.moves.new_tensor([3, 3, 3, 4]).view(1, 4, 1)
        self.max_speed = self.moves.new_tensor([1, 1, 1, 1.3]).view(1, 4, 1)
        radii = self.moves.new_tensor([.075, .075, .075, .05, .2, .2])
        self.touch_distance = radii[:4, None] + radii[None, :]
        self.interact = ~torch.eye(4, 6, dtype=torch.bool, device=device)
        # The scalar pair loop uses (+x) for the lower-index body at exact coincidence.
        signs = torch.where(torch.arange(4, device=device)[:, None] < torch.arange(6, device=device), 1, -1)
        self.coincident_normal = torch.stack((signs, torch.zeros_like(signs)), dim=-1)
        self.others = torch.tensor([[1, 2], [0, 2], [0, 1]], device=device)
        self.available = torch.ones(batch_size, 3, self.n_actions, dtype=torch.bool, device=device)

    def reset(self, layout="random"):
        if layout != "random":
            raise ValueError("PP uses standard random initial positions")
        positions = torch.rand((self.batch_size, 6, 2), generator=self.rng, device=self.device,
                               dtype=torch.float64) * 2 - 1
        positions[:, 4:] *= .9
        self.load_snapshot(positions, torch.zeros_like(positions[:, :4]))

    def load_snapshot(self, positions, velocity):
        self.positions = torch.as_tensor(positions, device=self.device, dtype=torch.float64).clone()
        self.velocity = torch.as_tensor(velocity, device=self.device, dtype=torch.float64).clone()
        if self.positions.shape != (self.batch_size, 6, 2) or self.velocity.shape != (self.batch_size, 4, 2):
            raise ValueError("Expected six positions and four mover velocities per board")
        self.steps = 0
        self.terminated = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.truncated = torch.zeros_like(self.terminated)

    def state(self):
        movers = torch.cat((self.positions[:, :4], self.velocity), dim=-1)
        return torch.cat((movers.flatten(1), self.positions[:, 4:].flatten(1)), dim=-1).float()

    def observations(self):
        predators = self.positions[:, :3]
        obstacles = self.positions[:, None, 4:] - predators[:, :, None]
        others = predators[:, self.others] - predators[:, :, None]
        return torch.cat((self.velocity[:, :3], predators, obstacles.flatten(2), others.flatten(2),
                          self.positions[:, 3:4] - predators, self.velocity[:, 3:4].expand(-1, 3, -1)),
                         dim=-1).float()

    def avail_actions(self):
        return self.available

    def _prey_action(self):
        prey = self.positions[:, 3:4]
        # The scalar policy receives a float32 observation and scores float64 offsets.
        offset = .2 * self.moves + .15 * self.velocity[:, 3:4].float()
        predators = (self.positions[:, :3] - prey).float()
        obstacles = (self.positions[:, 4:] - prey).float()
        clearance = torch.linalg.vector_norm(offset[:, :, None] - predators[:, None], dim=-1).amin(-1)
        obstacle_dist = torch.linalg.vector_norm(offset[:, :, None] - obstacles[:, None], dim=-1).amin(-1)
        edge = (prey.float() + offset).abs().sub(.9).clamp_min(0).sum(-1)
        return (clearance - 2 * (.28 - obstacle_dist).clamp_min(0) - 2 * edge).argmax(-1)

    def step(self, actions):
        if self.steps >= self.episode_limit:
            raise RuntimeError("Reset PP before stepping a finished episode")
        if actions.shape != (self.batch_size, self.n_agents) or actions.dtype != torch.long:
            raise ValueError("Provide one integer action per predator and board")
        self.last_prey_action = self._prey_action()
        joint = torch.cat((actions, self.last_prey_action[:, None]), dim=1)
        delta = self.positions[:, :4, None, :] - self.positions[:, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        normal = delta / distance.masked_fill(distance == 0, 1).unsqueeze(-1)
        normal = torch.where((distance == 0).unsqueeze(-1), self.coincident_normal, normal)
        penetration = torch.logaddexp(distance.new_zeros(()), (self.touch_distance - distance) / _CONTACT_MARGIN)
        contact = (_CONTACT_FORCE * _CONTACT_MARGIN * penetration * self.interact).unsqueeze(-1) * normal
        moves = self.moves.index_select(0, joint.flatten()).reshape(self.batch_size, 4, 2)
        force = self.acceleration * moves + contact.sum(2)
        velocity = (1 - _DAMPING) * self.velocity + _DT * force
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True).clamp_min(1e-15)
        self.velocity = velocity * (self.max_speed / speed).clamp_max(1)
        self.positions[:, :4] += _DT * self.velocity
        contacts = torch.linalg.vector_norm(self.positions[:, :3] - self.positions[:, 3:4], dim=-1)
        self.last_contacts = contacts < self.touch_distance[:3, 3]
        self.steps += 1
        if self.steps == self.episode_limit:
            self.truncated = torch.ones_like(self.truncated)
        return 10 * self.last_contacts.sum(1).float(), self.terminated, self.truncated


def check_parity(device):
    """Compare observations, dynamics, prey decisions, and contact rewards over full episodes."""
    games = [PredatorPreyWorld(seed=3100+i) for i in range(12)]
    try:
        for game in games:
            game.reset()
        # Exercise a successful contact and exact coincidence in addition to random boards.
        games[0].players[2].position = games[0].prey.position.copy()
        games[1].players[0].position = games[1].prey.position + [.13, 0]
        games[1].players[0].velocity[:] = [-.8, 0]
        world = TorchPredatorPrey(len(games), device)
        world.load_snapshot(np.array([[b.position for b in (*g.players, g.prey, *g.obstacles)] for g in games]),
                            np.array([[b.velocity for b in (*g.players, g.prey)] for g in games]))
        rng = np.random.default_rng(3100)
        tagged = 0
        for t in range(world.episode_limit):
            np.testing.assert_allclose(world.observations().cpu().numpy(), [g.observations() for g in games], atol=2e-6)
            actions = rng.integers(5, size=(len(games), 3))
            reward, terminal, truncated = world.step(torch.as_tensor(actions, device=device))
            expected = [g.step(a) for g, a in zip(games, actions)]
            np.testing.assert_array_equal(world.last_prey_action.cpu().numpy(), [g.last_prey_action for g in games])
            np.testing.assert_allclose(world.state().cpu().numpy(), [g.state() for g in games], atol=2e-6)
            np.testing.assert_array_equal(reward.cpu().numpy(), [r for r, _, _ in expected])
            np.testing.assert_array_equal(truncated.cpu().numpy(), [info["episode_limit"] for _, _, info in expected])
            assert not terminal.any().item()
            tagged += sum(info["contacts"] for _, _, info in expected)
        if not tagged:
            raise AssertionError("No successful tag exercised in PP parity check")
        return dict(boards=len(games), steps_per_board=world.episode_limit, contact_events=tagged, passed=True)
    finally:
        for game in games:
            game.close()
