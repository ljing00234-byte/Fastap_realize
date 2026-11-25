import unittest

import torch

from scripts.evaluate_lbf_probe_starts import coverage, scores


class ProbeComparisonTests(unittest.TestCase):
    def test_coverage_counts_openings_and_only_real_observed_moves(self):
        # One food is diagonally northwest; NORTH and WEST approach it.
        obs = [0,0,6, -1,-1,0, -1,-1,0, 1,1,1, -1,-1,0, -1,-1,0, -1,-1,0]
        local = dict(obs=torch.tensor(obs).expand(2, 2, 21),
                     available=torch.tensor([1,1,1,1,1,0], dtype=torch.bool).expand(2,2,6).clone())
        local['available'][1, 0, 5] = True
        batch = dict(filled=torch.tensor([[True, False], [True, True]]),
                     teammate_actions=torch.tensor([[[0,1], [0,1]], [[1,0], [2,0]]]),
                     teammate_mask=torch.tensor([[[0,1],[0,0]],[[1,0],[1,0]]], dtype=torch.bool))
        data = dict(members=['X','Y'], map_seeds=[5], batch=batch)
        counts = coverage(data, local, {'X':'SOCIAL1', 'Y':'BASELINE'})
        self.assertEqual(counts['all'], dict(episodes=2, actions=3, opportunities=2, approaches=1,
                                             opening_opportunities=1, opening_approaches=1))
        self.assertEqual(counts['SOCIAL1']['opening_approaches'], 1)

    def test_hard_pair_scores_are_independent_of_third_family_assignment(self):
        labels = {'A':'BASELINE', 'B':'BASELINE', 'C':'SOCIAL1', 'D':'SOCIAL1', 'E':'EGOISTIC'}
        result = scores(dict(A='c1', B='c1', C='c2', D='c2', E='c1'), labels)
        self.assertEqual(result['baseline_social1']['ari'], 1.)
        self.assertLess(result['all']['ari'], 1.)
