import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_scenarios import TeammateGroup, make_env
from fastap_scenarios.foraging import ForagingWorld
from fastap_scenarios.torch_foraging import TorchForaging
from fastap_scenarios.torch_lbf_ego import TorchLBFEgo, TorchMixedLBFEgo


class MemoryPolicy(torch.nn.Module):
    """Its preferred move depends on the original role, last action and recurrent memory."""

    def __init__(self, offset):
        super().__init__()
        self.register_buffer("choices", torch.arange(6, dtype=torch.float32))
        self.offset = offset

    def init_hidden(self):
        return self.choices.new_zeros(1, 1)

    def forward(self, inputs, hidden):
        previous = (inputs[:, 21:27] * self.choices).sum(1)
        preferred = (previous + 2 * inputs[:, -1] + hidden[:, 0] + self.offset) % 6
        return -(self.choices - preferred[:, None]).abs(), (hidden + 1) % 5


class EgoForagingTests(unittest.TestCase):
    def setUp(self):
        spec = AgentSpec("lbf", "team", 3, 21, 6, 1, True, True, True)
        self.pool = [FrozenAgent(spec, MemoryPolicy(offset)) for offset in (1, 3)]

    def test_frozen_role_control_and_recurrent_history_match_scalar_ego(self):
        for count in (1, 2):
            with self.subTest(n_teammates=count):
                self.check_role_and_history(count)

    def check_role_and_history(self, count):
        size = 5
        world = TorchLBFEgo(size, "cpu", teammates=self.pool, n_teammates=count)
        rng = np.random.default_rng(21)
        for episode in range(2):
            ids = (torch.arange(size) + episode) % len(self.pool)
            if count == 2:
                ids = torch.stack((ids, 1 - ids), dim=1)
            envs = []
            for row, row_ids in enumerate(ids.reshape(size, count).tolist()):
                group = TeammateGroup(str(row_ids), tuple(self.pool[i].policy_factory(2) for i in row_ids))
                env = make_env("lbf", control="ego", n_teammates=count, seed=300+row, teammate_groups=[group])
                self.addCleanup(env.close)
                env.reset()
                envs.append(env)
            world.load_snapshot(
                field=np.stack([e.world.field for e in envs]),
                positions=np.array([[p.position for p in e.world.players] for e in envs]),
                levels=np.array([[p.level for p in e.world.players] for e in envs]),
                teammate_ids=ids,
            )
            for _ in range(25):
                np.testing.assert_array_equal(world.observations(), [e.get_obs() for e in envs])
                np.testing.assert_array_equal(world.avail_actions(), [e.get_avail_actions() for e in envs])
                np.testing.assert_array_equal(world.state(), [e.get_state() for e in envs])
                actions = [[rng.choice(np.flatnonzero(mask)) for mask in e.get_avail_actions()] for e in envs]
                scalar = [e.step(a) for e, a in zip(envs, actions)]
                reward, terminated, truncated = world.step(torch.as_tensor(actions))
                np.testing.assert_array_equal(world.teammate_ids, ids)
                np.testing.assert_array_equal(world.last_joint_actions, [e.last_joint_actions for e in envs])
                np.testing.assert_allclose(reward, [r for r, _, _ in scalar])
                np.testing.assert_array_equal(terminated | truncated, [d for _, d, _ in scalar])
        self.assertEqual(world.sample_counts.tolist(), [5 * count, 5 * count])

    def test_training_samples_per_episode_and_keeps_board_rng_separate(self):
        a, b = [TorchLBFEgo(64, "cpu", teammates=self.pool, seed=8, teammate_seed=s) for s in (2, 3)]
        self.assertEqual(a.sample_counts.sum().item(), 0)
        for _ in range(2):
            a.reset()
            b.reset()
            torch.testing.assert_close(a.state(), b.state())
            self.assertFalse(torch.equal(a.teammate_ids, b.teammate_ids))
        self.assertEqual(a.sample_counts.sum().item(), 128)
        self.assertTrue((a.sample_counts > 0).all())

    def test_mixed_rosters_and_distinct_frozen_roles(self):
        world = TorchMixedLBFEgo(32, "cpu", teammates=self.pool)
        for _ in range(2):
            world.reset()
            first, second = world.worlds
            self.assertEqual(first.game.n_agents, 3)
            self.assertEqual(second.game.n_agents, 4)
            self.assertTrue((second.teammate_ids[:, 0] != second.teammate_ids[:, 1]).all())
            np.testing.assert_array_equal(second.game.field.amax((1, 2)),
                                          second.game.levels.sort().values[:, :3].sum(1))
            self.assertEqual(world.observations().shape, (32, 2, 21))
            self.assertEqual(world.state().shape, (32, 48))
            world.step(world.avail_actions().long().argmax(-1))
        self.assertEqual(world.roster_counts.tolist(), [32, 32])
        self.assertEqual(world.sample_counts.sum().item(), 96)
        with self.assertRaisesRegex(ValueError, "distinct"):
            second.reset(teammate_ids=torch.zeros(16, 2, dtype=torch.long))

    def test_four_player_cascading_collision_and_three_player_load(self):
        game = ForagingWorld(seed=402)
        self.addCleanup(game.close)
        game.reset(n_teammates=2)
        game.field[:] = 0
        game.field[3, 3] = 6
        game.field[1, 1] = 6
        game.initial_food_total = 12
        for player, cell, level in zip(game.players, [(3, 2), (2, 3), (3, 4), (3, 1)], [1, 2, 3, 3]):
            player.position, player.level = cell, level
        game._env._gen_valid_moves()
        world = TorchForaging(1, "cpu", n_agents=4)
        world.load_snapshot([game.field.tolist()], [[p.position for p in game.players]],
                            [[p.level for p in game.players]])
        # Player 0 attempts to enter the food, thereby also blocking player 3.
        for actions in ([4, 0, 0, 4], [5, 5, 0, 0], [5, 5, 5, 0]):
            expected_reward, _, _ = game.step(actions)
            reward, _, _ = world.step(torch.tensor([actions]))
            self.assertEqual(reward.item(), expected_reward)
            np.testing.assert_array_equal(world.state()[0], game.state())
            np.testing.assert_array_equal(world.observations()[0], game.observations())
        self.assertEqual(expected_reward, 0.5)
        # A stationary last player blocks a chain of three rightward moves.
        start = [(4, 1), (4, 2), (4, 3), (4, 4)]
        for player, position in zip(game.players, start):
            player.position = position
        game._env._gen_valid_moves()
        world.load_snapshot([game.field.tolist()], [start], [[p.level for p in game.players]],
                            initial_food_total=[game.initial_food_total], steps=[game.current_step])
        actions = [4, 4, 4, 0]
        game.step(actions)
        world.step(torch.tensor([actions]))
        np.testing.assert_array_equal(world.positions[0], start)
        np.testing.assert_array_equal(world.state()[0], game.state())
