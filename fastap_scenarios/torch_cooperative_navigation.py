"""Batched cooperative navigation with two ego slots, one teammate and four landmarks."""

import numpy as np
import torch
from torch.nn import functional as F

from .cooperative_navigation import (
    AGENT_ACCELERATION, AGENT_RADIUS, DEFAULT_COLLISION_PENALTY, CooperativeNavigationWorld,
)
from .particle import MOVES, _DAMPING, _DT
from .torch_particle import ParticleContacts


class TorchCooperativeNavigation:
    scene_key = "cn"
    n_agents = 3
    n_actions = CooperativeNavigationWorld.n_actions
    obs_shape = CooperativeNavigationWorld.obs_shape
    state_shape = CooperativeNavigationWorld.state_shape
    episode_limit = 25

    def __init__(self, batch_size, device, seed=0):
        if batch_size < 1:
            raise ValueError("CN requires at least one board")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.moves = torch.as_tensor(MOVES, device=self.device)
        self.contacts = ParticleContacts(self.moves.new_full((3,), AGENT_RADIUS), movers=3)
        self.others = torch.tensor([[1, 2], [0, 2], [0, 1]], device=self.device)
        self.pair_i, self.pair_j = torch.triu_indices(3, 3, offset=1, device=self.device)
        self.available = torch.ones(batch_size, 3, self.n_actions, dtype=torch.bool, device=self.device)

    def reset(self, layout="random"):
        if layout != "random":
            raise ValueError("CN uses random agent and landmark positions")
        coords = torch.rand((self.batch_size, 7, 2), device=self.device, dtype=torch.float64,
                            generator=self.rng) * 2 - 1
        self.load_snapshot(coords[:, 4:], torch.zeros_like(coords[:, 4:]), coords[:, :4])

    def load_snapshot(self, positions, velocity, landmarks):
        def read(value, slots):
            tensor = torch.as_tensor(value, device=self.device, dtype=torch.float64)
            if tensor.shape != (self.batch_size, slots, 2):
                raise ValueError(f"Expected {slots} two-dimensional vectors per board")
            return tensor.clone()
        self.positions = read(positions, 3)
        self.velocity = read(velocity, 3)
        self.landmarks = read(landmarks, 4)
        self.steps = 0
        self.terminated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros_like(self.terminated)

    def observations(self):
        relative = self.landmarks[:, None] - self.positions[:, :, None]
        peers = self.positions[:, self.others] - self.positions[:, :, None]
        peers = F.pad(peers, (0, 1), value=1).flatten(2)
        return torch.cat((self.velocity, self.positions, relative.flatten(2), F.pad(peers, (0, 3))), dim=-1).float()

    def state(self):
        movers = F.pad(torch.cat((self.positions, self.velocity), dim=-1), (0, 1), value=1)
        return torch.cat((F.pad(movers.flatten(1), (0, 5)), self.landmarks.flatten(1)), dim=-1).float()

    def avail_actions(self):
        return self.available

    def step(self, actions):
        if self.steps >= self.episode_limit:
            raise RuntimeError("Reset CN before stepping a finished episode")
        if actions.shape != (self.batch_size, self.n_agents) or actions.dtype != torch.long:
            raise ValueError("Provide one integer action per active agent and board")
        moves = self.moves.index_select(0, actions.flatten()).reshape(self.batch_size, 3, 2)
        force = AGENT_ACCELERATION * moves + self.contacts.forces(self.positions)
        self.velocity = (1 - _DAMPING) * self.velocity + _DT * force
        self.positions = self.positions + _DT * self.velocity
        distances = torch.linalg.vector_norm(self.landmarks[:, :, None] - self.positions[:, None], dim=-1)
        self.distance_sum = distances.amin(-1).sum(-1)
        pairs = self.positions[:, self.pair_i] - self.positions[:, self.pair_j]
        self.collisions = (torch.linalg.vector_norm(pairs, dim=-1) < 2 * AGENT_RADIUS).sum(-1)
        self.steps += 1
        if self.steps == self.episode_limit:
            self.truncated = torch.ones_like(self.truncated)
        reward = -self.distance_sum - DEFAULT_COLLISION_PENALTY * self.collisions
        return reward.float(), self.terminated, self.truncated


def check_parity(device):
    games = [CooperativeNavigationWorld(seed=4100+i) for i in range(12)]
    try:
        for game in games:
            game.reset(n_teammates=1)
        games[0].agents[1].position[:] = games[0].agents[0].position
        world = TorchCooperativeNavigation(len(games), device)
        world.load_snapshot(np.array([[a.position for a in g.agents] for g in games]),
                            np.array([[a.velocity for a in g.agents] for g in games]),
                            np.array([[l.position for l in g.landmarks] for g in games]))
        rng = np.random.default_rng(4100)
        hits = 0
        for step in range(world.episode_limit + 1):
            np.testing.assert_allclose(world.observations().cpu().numpy(), [g.observations() for g in games], atol=2e-6)
            np.testing.assert_allclose(world.state().cpu().numpy(), [g.state() for g in games], atol=2e-6)
            np.testing.assert_array_equal(world.avail_actions().cpu().numpy(), [g.avail_actions() for g in games])
            if step == world.episode_limit:
                break
            actions = rng.integers(world.n_actions, size=(len(games), 3))
            reward, terminal, truncated = world.step(torch.as_tensor(actions, device=device))
            results = [g.step(a) for g, a in zip(games, actions)]
            np.testing.assert_allclose(reward.cpu().numpy(), [r for r, _, _ in results], atol=2e-6)
            np.testing.assert_allclose(world.distance_sum.cpu().numpy(), [i["landmark_distance_sum"] for _, _, i in results], atol=2e-6)
            np.testing.assert_array_equal(world.collisions.cpu().numpy(), [i["agent_collisions"] for _, _, i in results])
            np.testing.assert_array_equal(truncated.cpu().numpy(), [d for _, d, _ in results])
            if terminal.any().item():
                raise AssertionError("Unexpected early termination")
            hits += int(world.collisions.sum().item())
        if not hits:
            raise AssertionError("No collision tested")
        return dict(boards=len(games), steps_per_board=world.episode_limit, collision_events=hits, passed=True)
    finally:
        for game in games:
            game.close()
