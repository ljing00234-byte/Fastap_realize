import unittest
from unittest.mock import patch

import numpy as np

from fastap_scenarios import SCENARIOS, TeammateGroup
from fastap_scenarios.fastap_rollout import FastapRolloutEnv
from fastap_scenarios.policies import stay


class FastapRolloutTests(unittest.TestCase):
    def test_transition_targets_and_raw_reward_match_the_existing_game(self):
        for scene, scenario in SCENARIOS.items():
            for count in scenario.teammate_counts:
                with self.subTest(scene=scene, count=count):
                    groups = [TeammateGroup("fixed", (lambda: stay,) * count)]
                    env = FastapRolloutEnv(scene, groups, seed=33)
                    reference = FastapRolloutEnv(scene, groups, seed=33)
                    self.addCleanup(env.close)
                    self.addCleanup(reference.close)
                    obs, state, avail = env.reset(0)
                    reference.reset(0)
                    self.assertEqual(obs.shape, (2, env.obs_shape))
                    self.assertEqual(state.shape, (env.state_shape,))
                    self.assertEqual(avail.shape, (2, env.n_actions))
                    for t in range(env.episode_limit):
                        before = reference.world.observations()[2:]
                        actions = [int(np.flatnonzero(mask)[-1]) for mask in avail]
                        reward, done, info = env.step(actions)
                        raw, terminal, diagnostics = reference.world.step(actions + [0] * count)
                        self.assertEqual((reward, done, info["episode_limit"]),
                                         (raw, terminal, diagnostics["episode_limit"]))
                        self.assertEqual(info["group_index"], 0)
                        np.testing.assert_array_equal(info["teammate_obs"][:count], before)
                        np.testing.assert_array_equal(info["teammate_obs"][count:], 0)
                        self.assertEqual(info["teammate_actions"].tolist(),
                                         [0] * count + [-1] * (env.max_teammates - count))
                        obs, state, avail = env.observe()
                        np.testing.assert_array_equal(state, reference.world.state())
                        if done:
                            break
                    self.assertTrue(done)
                    self.assertEqual(len(env.events), 1)

    def test_redraw_records_label_of_acting_group_on_join_and_leave(self):
        for scene in ("cn", "lbf"):
            with self.subTest(scene=scene):
                groups = [TeammateGroup("solo", (lambda: stay,)), TeammateGroup("duo", (lambda: stay,) * 2)]
                env = FastapRolloutEnv(scene, groups, seed=17)
                self.addCleanup(env.close)
                env.reset(0, switch_interval=(1, 1))
                with patch.object(env._stream, "group_rng") as group_rng:
                    group_rng.integers.side_effect = (1, 0)
                    for old, new in ((0, 1), (1, 0)):
                        before = env.world.observations()[2:]
                        _, _, info = env.step([0, 0])
                        self.assertEqual(info["group_index"], old)
                        self.assertEqual(env.group_index, new)
                        np.testing.assert_array_equal(info["teammate_obs"][:len(before)], before)
                        self.assertEqual(info["teammate_actions"].tolist(), [0, -1] if old == 0 else [0, 0])
                        self.assertEqual(env.n_teammates, len(groups[new].policies))
                self.assertEqual([event["step"] for event in env.events], [0, 1, 2])
