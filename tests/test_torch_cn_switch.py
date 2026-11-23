import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_scenarios.cooperative_navigation import CooperativeNavigationWorld
from fastap_scenarios.torch_cn_switch import TorchSwitchingCNEgo


class CNSwitchWorldTests(unittest.TestCase):
    def test_group_changes_preserve_scalar_physics_and_recurrent_history(self):
        spec = AgentSpec("cn", "team", 3, 21, 5, 8, True, True, True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            pool = [FrozenAgent(spec, spec.build_network()) for _ in range(3)]
        world = TorchSwitchingCNEgo(3, "cpu", teammates=pool)
        games = [CooperativeNavigationWorld(seed=30+i) for i in range(3)]
        groups = [[0, -1], [0, 1], [2, -1]]
        for game, group in zip(games, groups):
            self.addCleanup(game.close)
            game.reset(n_teammates=1 + int(group[1] >= 0))
            # The inactive slot is also stored at the origin. It must not collide
            # with this body or count toward a landmark's nearest-agent distance.
            game.agents[0].position[:] = 0.
            game.landmarks[0].position[:] = 0.

        def slots(game, key):
            values = [getattr(a, key) for a in game.agents]
            return values + [np.zeros(2)] * (4 - len(values))

        def controllers(group):
            return [pool[i].policy_factory(2)() for i in group if i >= 0]

        world.load_snapshot(positions=np.array([slots(g, "position") for g in games]),
                            velocity=np.array([slots(g, "velocity") for g in games]),
                            landmarks=np.array([[lm.position for lm in g.landmarks] for g in games]),
                            teammate_ids=torch.tensor(groups))
        policies = [controllers(group) for group in groups]
        schedule = {2: [[0, 1], [0, -1], [2, -1]],
                    5: [[0, 2], [1, -1], [2, 0]],
                    9: [[2, -1], [1, 0], [2, 0]],
                    15: [[2, 1], [1, 0], [0, -1]]}
        rng = np.random.default_rng(28)
        saw_collision = False
        for t in range(world.episode_limit):
            if t in schedule:
                new_groups = schedule[t]
                spawning = np.zeros((len(games), 2))
                for row, (g, old, new) in enumerate(zip(games, groups, new_groups)):
                    if old != new:
                        policies[row] = controllers(new)
                    g.set_teammate_count(1 + int(new[1] >= 0))
                    if new[1] >= 0 and old[1] < 0:
                        if t == 2:
                            g.agents[3].position[:] = g.agents[0].position
                        spawning[row] = g.agents[3].position
                changed = world.apply_teammates(torch.tensor(new_groups), torch.from_numpy(spawning))
                self.assertEqual(changed.tolist(), [old != new for old, new in zip(groups, new_groups)])
                groups = new_groups
            np.testing.assert_allclose(world.observations(), [g.observations()[:2] for g in games], atol=2e-6)
            np.testing.assert_allclose(world.state(), [g.state() for g in games], atol=2e-6)
            actions = rng.integers(5, size=(len(games), 2))
            rewards, _, truncated = world.step(torch.from_numpy(actions))
            for row, g in enumerate(games):
                obs, mask = g.observations(), g.avail_actions()
                mates = [policy(obs[2+j], mask[2+j], rng) for j, policy in enumerate(policies[row])]
                joint = [*actions[row], *mates]
                self.assertEqual(world.last_joint_actions[row, :len(joint)].tolist(), joint)
                reward, done, info = g.step(joint)
                self.assertAlmostEqual(rewards[row].item(), reward, places=5)
                self.assertAlmostEqual(world.game.distance_sum[row].item(), info["landmark_distance_sum"], places=9)
                self.assertEqual(world.game.collisions[row].item(), info["agent_collisions"])
                self.assertEqual(truncated[row].item(), done)
                saw_collision |= bool(info["agent_collisions"])
                for j, policy in enumerate(policies[row]):
                    torch.testing.assert_close(world._hidden[2*row+j], policy.hidden[0], rtol=1e-5, atol=2e-6)
                    torch.testing.assert_close(world._previous[2*row+j], policy.last_action)
        self.assertTrue(saw_collision)
        final_state = world.state().clone()
        self.assertFalse(world.apply_teammates(torch.tensor([[0, -1]] * len(games))).any())
        torch.testing.assert_close(final_state, world.state())
