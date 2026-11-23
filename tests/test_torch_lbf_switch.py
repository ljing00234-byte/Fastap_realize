import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_scenarios.foraging import ForagingWorld
from fastap_scenarios.torch_lbf_switch import TorchSwitchingLBFEgo


class OpenLBFTests(unittest.TestCase):
    def setUp(self):
        spec = AgentSpec("lbf", "team", 3, 21, 6, 8, True, True, True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            self.pool = [FrozenAgent(spec, spec.build_network()) for _ in range(3)]

    def test_episode_changes_match_scalar_physics_and_group_memory(self):
        size = 3
        env = TorchSwitchingLBFEgo(size, "cpu", teammates=self.pool, seed=312)
        games = [ForagingWorld(seed=910 + row) for row in range(size)]
        for game in games:
            self.addCleanup(game.close)
            game.reset(n_teammates=1)
        ids = [[0, -1], [1, -1], [2, -1]]
        env.load_snapshot(teammate_ids=torch.tensor(ids),
                          fourth_levels=[g._levels[3] for g in games],
                          field=np.stack([g.field for g in games]),
                          positions=np.array([[p.position for p in g.players] for g in games]),
                          levels=np.array([[p.level for p in g.players] for g in games]))
        schedule = {
            1: [[0, 1], [1, -1], [0, -1]],
            3: [[0, 2], [2, 0], [0, -1]],
            5: [[0, -1], [2, 0], [0, 2]],
            7: [[0, 1], [1, -1], [2, 0]],
        }

        def controllers(group):
            return [self.pool[i].policy_factory(2)() for i in group if i >= 0]

        policies = [controllers(group) for group in ids]
        for t in range(25):
            if t in schedule:
                next_groups = schedule[t]
                ranks = torch.ones(size, 36)
                for row, (game, before, after) in enumerate(zip(games, ids, next_groups)):
                    if before != after:
                        policies[row] = controllers(after)
                    game.set_teammate_count(1 + int(after[1] >= 0))
                    if after[1] >= 0:
                        r, c = game.players[3].position
                        ranks[row, 6 * r + c] = 0
                changed = env.apply_teammates(torch.tensor(next_groups), ranks)
                self.assertEqual(changed.tolist(), [a != b for a, b in zip(ids, next_groups)])
                ids = next_groups
            np.testing.assert_array_equal(env.state(), np.stack([g.state() for g in games]))
            np.testing.assert_array_equal(env.observations(), np.stack([g.observations()[:2] for g in games]))
            np.testing.assert_array_equal(env.avail_actions(), np.stack([g.avail_actions()[:2] for g in games]))
            ego = torch.zeros(size, 2, dtype=torch.long)
            reward, finished, timeout = env.step(ego)
            for row, game in enumerate(games):
                obs, masks = game.observations(), game.avail_actions()
                actions = [p(obs[2 + j], masks[2 + j], None) for j, p in enumerate(policies[row])]
                scalar_reward, done, _ = game.step([0, 0, *actions])
                self.assertEqual(env.last_joint_actions[row, :2 + len(actions)].tolist(), [0, 0, *actions])
                self.assertEqual(done, (finished | timeout)[row].item())
                self.assertAlmostEqual(scalar_reward, reward[row].item(), places=6)
                for j, policy in enumerate(policies[row]):
                    torch.testing.assert_close(env._hidden[2 * row + j], policy.hidden[0], rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(env._previous[2 * row + j], policy.last_action)
        state = env.state().clone()
        self.assertFalse(env.apply_teammates(torch.tensor([[1, -1]] * size)).any())
        torch.testing.assert_close(state, env.state())

    def test_native_reset_uses_the_starting_rosters_food_level(self):
        env = TorchSwitchingLBFEgo(2, "cpu", teammates=self.pool, seed=9)
        env.reset(teammate_ids=torch.tensor([[1, -1], [2, 0]]))
        self.assertEqual(env.game.field[0].max().item(), env.game.levels[0, :3].sum().item())
        self.assertEqual(env.game.field[1].max().item(), env.game.levels[1].sort().values[:3].sum().item())
        self.assertEqual(env.state()[0, -3:].tolist(), [-1, -1, 0])
        initial = env.game.initial_food_total.clone()
        env.apply_teammates(torch.tensor([[1, 2], [1, -1]]))
        torch.testing.assert_close(env.game.initial_food_total, initial)
