import unittest

import numpy as np
import torch

from fastap_scenarios.policies import stay
from scripts.lbf_rule_probes import RULES, action_probabilities, collect_episode


class LbfRuleProbeTests(unittest.TestCase):
    def test_rules_distinguish_loading_exploration_and_approach(self):
        obs = np.array([[1, 0, 6], [0, 2, 6], [-1, -1, 0],
                        [1, 1, 2], [2, 2, 1], [1, 2, 3], [-1, -1, 0]]).ravel()
        # West cell has food, east cell has another player. NORTH and SOUTH are open.
        mask = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
        self.assertEqual(action_probabilities('BASELINE', obs, mask)[5], 1.)
        np.testing.assert_array_equal(action_probabilities('EGOISTIC', obs, mask), [0, .5, .5, 0, 0, 0])
        self.assertEqual(action_probabilities('SOCIAL1', obs, mask)[1], 1.)
        for rule in RULES:
            self.assertEqual(action_probabilities(rule, obs, mask)[~mask].sum(), 0.)

    def test_equal_target_ties_and_blocked_exploration(self):
        obs = np.array([[0, 0, 4], [0, 2, 4], [-1, -1, 0],
                        [1, 1, 2], [-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]).ravel()
        mask = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
        np.testing.assert_allclose(action_probabilities('SOCIAL1', obs, mask), [0, .5, 0, .25, .25, 0])
        mask[1:] = False
        for rule in RULES:
            self.assertEqual(action_probabilities(rule, obs, mask)[0], 1.)

    def test_matched_board_independent_replica_streams_and_reproducibility(self):
        egos = (lambda: stay, lambda: stay)
        runs = [collect_episode('EGOISTIC', i, 6830000, egos) for i in (0, 0, 1)]
        first, local = runs[0]
        for key in first:
            self.assertTrue(torch.equal(first[key], runs[1][0][key]))
        self.assertTrue(torch.equal(first['state'][0], runs[2][0]['state'][0]))
        self.assertFalse(torch.equal(first['teammate_actions'], runs[2][0]['teammate_actions']))
        self.assertEqual(local['obs'].shape, (25, 21))
        valid = first['filled']
        indices = first['teammate_actions'][valid, 0]
        self.assertTrue(local['available'][valid].gather(1, indices[:, None]).all())
        self.assertFalse(first['teammate_mask'][:, 1].any())


if __name__ == '__main__':
    unittest.main()
