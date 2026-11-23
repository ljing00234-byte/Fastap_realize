import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_scenarios import TeammateGroup, make_env
from fastap_scenarios.torch_cn_ego import TorchCNEgo


class CooperativeNavigationEgoTests(unittest.TestCase):
    def test_full_episode_matches_scalar_role_actions_state_and_team_return(self):
        spec = AgentSpec("cn", "team", 3, 21, 5, 8, True, True, True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            pool = [FrozenAgent(spec, spec.build_network()) for _ in range(2)]
        world = TorchCNEgo(len(pool), "cpu", teammates=pool, seed=20)
        scalar = []
        for i, member in enumerate(pool):
            group = TeammateGroup(str(i), (member.policy_factory(2),))
            env = make_env("cn", control="ego", seed=21+i, n_teammates=1, teammate_groups=[group])
            self.addCleanup(env.close)
            env.reset()
            scalar.append(env)
        world.load_snapshot(
            positions=np.array([[a.position for a in env.world.agents] for env in scalar]),
            velocity=np.array([[a.velocity for a in env.world.agents] for env in scalar]),
            landmarks=np.array([[p.position for p in env.world.landmarks] for env in scalar]),
            teammate_ids=torch.arange(len(pool)),
        )
        rng = np.random.default_rng(23)
        for step in range(world.episode_limit):
            np.testing.assert_allclose(world.observations(), [env.get_obs() for env in scalar], atol=2e-6)
            actions = rng.integers(world.n_actions, size=(len(scalar), 2))
            result = [env.step(a) for env, a in zip(scalar, actions)]
            reward, terminal, truncated = world.step(torch.from_numpy(actions))
            np.testing.assert_array_equal(world.last_joint_actions, [env.last_joint_actions for env in scalar])
            np.testing.assert_allclose(world.state(), [env.get_state() for env in scalar], atol=2e-6)
            np.testing.assert_allclose(reward, [r for r, _, _ in result], atol=2e-6)
            self.assertFalse(terminal.any())
            self.assertTrue((truncated == (step == world.episode_limit - 1)).all())
        self.assertEqual(world.sample_counts.tolist(), [1, 1])
        self.assertEqual(world.roster_counts.tolist(), [2])
