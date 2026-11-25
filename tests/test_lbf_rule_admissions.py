import unittest

import torch

from scripts.audit_lbf_rule_admissions import classify_vectors, decision_errors, pair_errors, summarize


class RuleAdmissionsTests(unittest.TestCase):
    def test_count_split_and_merge_separately_and_compare_seeds_on_matching_blocks(self):
        labels = dict(A='BASELINE', B='BASELINE', C='SOCIAL1', D='SOCIAL1')
        mixed = dict(A='C1', B='C2', C='C1', D='C1')
        result = pair_errors(mixed, labels)
        self.assertEqual((result['split'], result['same'], result['merged'], result['different']), (1, 2, 2, 4))
        rows = [dict(seed=seed, block=i, assignments=a)
                for seed in (7, 17) for i, a in enumerate([labels, mixed])]
        report = summarize(rows, labels)['all']
        self.assertEqual(report['split'], 2)
        self.assertEqual(report['merged'], 4)
        self.assertEqual(report['between_seed_ari'], 1.)

    def test_encoder_readout_excludes_query_identity(self):
        rules = ['BASELINE', 'BASELINE', 'SOCIAL1', 'SOCIAL1']
        # Each identity is copied across maps, but its same-rule *other* identity is far away.
        x = torch.tensor([0., 10., 1., 11.]).reshape(1, 4, 1)
        result = classify_vectors(x, x, rules, 1, ('BASELINE', 'SOCIAL1'))
        self.assertEqual((result['correct'], result['ties'], result['total']), (0, 0, 4))
        result = classify_vectors(x*0, x*0, rules, 1, ('BASELINE', 'SOCIAL1'))
        self.assertEqual((result['correct'], result['ties']), (0, 4))

    def test_after_scoring_choose_lowest_error_counterfactual(self):
        # The newly admitted member splits from B, and has no disagreement with A.
        labels = dict(A='SOCIAL1', B='BASELINE', X='BASELINE')
        candidates = [dict(cluster=name, log_prior=0., log_score=v)
                      for name, v in [('C1', -0.6), ('C2', -1.), ('C3', -0.3)]]
        row = dict(episode=0, step=2, action=1, opportunity='other', map_seed=15, logp=[-.6, -1., -.3])
        events = [dict(member='X', previous=dict(A='C1', B='C2'),
                       trace=dict(winner='C3', candidates=candidates, episodes=1, rows=[row]))]
        case, = decision_errors(events, labels)
        self.assertEqual(case['alternative'], 'C2')
        self.assertEqual(case['split_from'], ['B'])
        self.assertEqual(case['merged_with'], [])
        self.assertAlmostEqual(case['evidence_margin'], .7)
        self.assertEqual(case['strongest'][0]['map_seed'], 15)
