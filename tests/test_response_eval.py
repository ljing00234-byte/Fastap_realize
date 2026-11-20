from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fastap_methods.response_eval import evaluate_pair
from fastap_scenarios import make_world


class ResponseEvaluationTests(unittest.TestCase):
    def test_paired_maps_original_roles_and_independent_episode_state(self):
        checkpoints = {"team": ("team", 3), "ego": ("ego", 2)}
        roles = []
        first_views = []

        def load(path, *, device):
            self.assertEqual(device, "cpu")
            control, n_agents = checkpoints[path]

            def policy_factory(slot):
                def fresh():
                    roles.append((path, slot))
                    seen = False

                    def act(observation, mask, rng):
                        nonlocal seen
                        if not seen and slot == 0:
                            first_views.append(observation.copy())
                        seen = True
                        return int(np.flatnonzero(mask)[-1])
                    return act
                return fresh

            return SimpleNamespace(spec=SimpleNamespace(scene_key="lbf", control=control, n_agents=n_agents),
                                   policy_factory=policy_factory)

        seeds = [77, 78]
        with patch("fastap_methods.response_eval.load_frozen_agent", side_effect=load):
            original = evaluate_pair("team", "team", seeds)
            self.assertEqual(roles, [("team", 0), ("team", 1), ("team", 2)] * 2)
            roles.clear()
            learned = evaluate_pair("ego", "team", seeds)
            self.assertEqual(roles, [("ego", 0), ("ego", 1), ("team", 2)] * 2)
            roles.clear()
            replacement = evaluate_pair("ego", "team", seeds, random_teammate=True)
            self.assertEqual(roles, [("ego", 0), ("ego", 1)] * 2)
            self.assertEqual(replacement, evaluate_pair("ego", "team", seeds, random_teammate=True))
        self.assertEqual(original, learned)
        self.assertEqual(learned["seeds"], seeds)
        for tick, seed in enumerate(seeds):
            world = make_world("lbf", seed=seed)
            try:
                world.reset(n_teammates=1)
                for start in range(0, len(first_views), len(seeds)):
                    np.testing.assert_array_equal(world.observations()[0], first_views[start + tick])
            finally:
                world.close()

    def test_statistics_and_invalid_teammate_roster(self):
        team = SimpleNamespace(spec=SimpleNamespace(scene_key="lbf", control="team", n_agents=3),
                               policy_factory=lambda slot: lambda: None)
        with patch("fastap_methods.response_eval.load_frozen_agent", return_value=team):
            with patch("fastap_methods.response_eval._episode_return", side_effect=[0.0, 2 / 3]):
                metrics = evaluate_pair("team", "team", [10, 11])
            self.assertEqual(metrics["n_episodes"], 2)
            self.assertEqual(metrics["success_rate"], 0.5)
            self.assertAlmostEqual(metrics["mean_return"], 1 / 3)
            self.assertAlmostEqual(metrics["return_se"], 1 / 3)
            self.assertEqual(metrics["episode_returns"], [0.0, 2 / 3])
            team.spec.control, team.spec.n_agents = "ego", 2
            with self.assertRaisesRegex(ValueError, "frozen teammate"):
                evaluate_pair("ego", "ego", [1, 2])
        with self.assertRaisesRegex(ValueError, "distinct"):
            evaluate_pair("unused", "unused", [7, 7])
