import unittest

import numpy as np

from scripts.lbf_rule_positive_control import RULES, average_linkage, behavior_events, controls, reveal


class RulePositiveControlTests(unittest.TestCase):
    def test_behavior_features_use_only_observed_opportunities_and_legal_moves(self):
        obs = np.zeros(21)
        obs[:3] = [0, 0, 5]
        obs[9:12] = [1, 1, 1]
        avail = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
        for action, expected in [(1, [0, 1]), (2, [0, 0])]:
            success, opportunity = behavior_events(obs, avail, action)
            np.testing.assert_equal(success, expected)
            np.testing.assert_equal(opportunity, [0, 1])
        avail[[2, 4]] = False
        _, opportunity = behavior_events(obs, avail, 1)
        np.testing.assert_equal(opportunity, [0, 0])
        obs[9:11] = [0, 1]
        avail[5] = True
        success, opportunity = behavior_events(obs, avail, 5)
        np.testing.assert_equal(success, [1, 0])
        np.testing.assert_equal(opportunity, [1, 0])

    def test_known_k_merge_and_oracle_ties_before_revealing_types(self):
        features = np.array([[.1, .5], [.9, .95], [.1, .48], [.9, .25], [.91, .96], [.9, .28]])
        self.assertEqual(average_linkage(features, 3), [(0, 2), (1, 4), (3, 5)])
        members = ['x', 'y', 'z']
        measured = dict(counts=np.zeros((1, 3, 2, 2), dtype=int),
                        likelihoods=np.array([[[-3., -3., -np.inf], [-np.inf, -1., -4.], [-3., -9., -1.]]]),
                        steps=np.ones((1, 3), dtype=int), different=np.zeros((1, 3, 3, 3), dtype=int),
                        tv_sum=np.zeros((1, 3, 3, 3)))
        result = controls(measured, members, slice(None))
        self.assertEqual(result['known_policy']['members']['x']['best_candidates'], list(RULES[:2]))
        reveal(result, dict(zip(members, RULES)), members)
        self.assertEqual(result['known_policy']['unique_correct'], 2)
        self.assertEqual(result['blind']['partition_vs_truth']['ari'], 1.)
