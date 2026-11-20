from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_methods.cooperative_navigation_eval import _rollout
from fastap_methods.pool_crossplay import evaluate_pool
from fastap_scenarios import TeammateGroup


class NavigationPoolTests(unittest.TestCase):
    def test_crossplay_matches_independent_ego_rollouts_for_every_pairing(self):
        spec = AgentSpec(scene_key="cn", control="team", n_agents=3, obs_shape=21, n_actions=5,
                         hidden_dim=16, use_rnn=True, obs_last_action=True, obs_agent_id=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            agents = [FrozenAgent(spec, spec.build_network()) for _ in range(2)]
        seeds = range(2_010_000, 2_010_008)
        members = [dict(id=name, path=name) for name in ("T1", "T2")]
        with patch("fastap_methods.pool_crossplay.load_frozen_agent", side_effect=agents):
            report = evaluate_pool(Path("."), members, seeds.start, len(seeds))

        self.assertEqual(report["reference_decisions"], len(seeds) * 25)
        self.assertGreater(report["action_disagreement"][0][1], 0)
        for row, ego in enumerate(agents):
            for column, teammate in enumerate(agents):
                group = TeammateGroup("test", (teammate.policy_factory(2),))
                returns, distances, collisions = np.mean([_rollout(seed, ego, group) for seed in seeds], axis=0)
                score = report["crossplay"][row][column]
                self.assertAlmostEqual(score["mean_return"], returns, places=4)
                self.assertAlmostEqual(score["mean_landmark_distance_sum_per_step"], distances, places=6)
                self.assertEqual(score["mean_pair_collisions_per_episode"], collisions)
                self.assertAlmostEqual(score["mean_return"], -25 * distances - collisions, places=4)
