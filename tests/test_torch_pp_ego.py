import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_scenarios import TeammateGroup, make_env
from fastap_scenarios.torch_pp_ego import TorchPPEgo


class MemoryHunter(torch.nn.Module):
    """Choose actions using the exported role, local history and policy identity."""

    def __init__(self, offset):
        super().__init__()
        self.register_buffer("actions", torch.arange(5, dtype=torch.float32))
        self.offset = offset

    def init_hidden(self):
        return self.actions.new_zeros(1, 1)

    def forward(self, inputs, hidden):
        previous = (inputs[:, 16:21] * self.actions).sum(1)
        preferred = (previous + 2 * inputs[:, -1] + hidden[:, 0] + self.offset) % 5
        return -(self.actions - preferred[:, None]).abs(), (hidden + 1) % 3


class PredatorPreyEgoTests(unittest.TestCase):
    def setUp(self):
        spec = AgentSpec("pp", "team", 3, 16, 5, 1, True, True, True)
        self.pool = [FrozenAgent(spec, MemoryHunter(i)) for i in (1, 3)]

    def test_frozen_hunter_and_team_scores_match_scalar_control_over_two_episodes(self):
        size = 4
        world = TorchPPEgo(size, "cpu", teammates=self.pool)
        rng = np.random.default_rng(21)
        for episode in range(2):
            ids = (torch.arange(size) + episode) % len(self.pool)
            scalar = []
            for row, member in enumerate(ids.tolist()):
                group = TeammateGroup(str(member), (self.pool[member].policy_factory(2),))
                env = make_env("pp", control="ego", seed=20 + row, n_teammates=1, teammate_groups=[group])
                self.addCleanup(env.close)
                env.reset()
                scalar.append(env)
            # Guarantee a contact so an accidental omission from the score is observable.
            game = scalar[0].world
            game.prey.position[:] = (0, 0)
            game.obstacles[0].position[:], game.obstacles[1].position[:] = (-.8, .8), (.8, -.8)
            for p, direction in zip(game.players, np.array([[1, 0], [-1, 0], [0, 1]])):
                p.position[:], p.velocity[:] = .14 * direction, -.8 * direction
            world.load_snapshot(
                positions=np.array([[b.position for b in (*e.world.players, e.world.prey, *e.world.obstacles)]
                                    for e in scalar]),
                velocity=np.array([[b.velocity for b in (*e.world.players, e.world.prey)] for e in scalar]),
                teammate_ids=ids)
            total_contacts = 0
            for step in range(25):
                np.testing.assert_allclose(world.observations(), [e.get_obs() for e in scalar], atol=2e-6)
                np.testing.assert_array_equal(world.avail_actions(), [e.get_avail_actions() for e in scalar])
                actions = rng.integers(5, size=(size, 2))
                results = [e.step(a) for e, a in zip(scalar, actions)]
                reward, terminal, truncated = world.step(torch.from_numpy(actions))
                np.testing.assert_array_equal(world.last_joint_actions, [e.last_joint_actions for e in scalar])
                np.testing.assert_allclose(world.state(), [e.get_state() for e in scalar], atol=2e-6)
                np.testing.assert_array_equal(world.teammate_ids, ids)
                np.testing.assert_array_equal(reward, [r for r, _, _ in results])
                np.testing.assert_array_equal(reward, 10 * world.game.last_contacts.sum(1))
                self.assertFalse(terminal.any())
                self.assertTrue((truncated == (step == 24)).all())
                total_contacts += world.game.last_contacts.sum().item()
            self.assertGreater(total_contacts, 0)
        self.assertEqual(world.sample_counts.tolist(), [4, 4])
        self.assertEqual(world.roster_counts.tolist(), [8])

    def test_teammate_sampling_does_not_change_initial_positions(self):
        a, b = [TorchPPEgo(32, "cpu", teammates=self.pool, seed=9, teammate_seed=s) for s in (7, 8)]
        a.reset()
        b.reset()
        torch.testing.assert_close(a.state(), b.state())
        self.assertFalse(torch.equal(a.teammate_ids, b.teammate_ids))
        self.assertEqual(a.roster_counts.tolist(), [32])
