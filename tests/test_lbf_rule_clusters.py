import unittest

import torch

from scripts.evaluate_lbf_crp_prediction import FIELDS
from scripts.validate_lbf_rule_clusters import member_batch, partition_score


class BlindRulePartitionTests(unittest.TestCase):
    def test_cluster_names_are_irrelevant_and_exact_null_counts_partitions_once(self):
        truth = {str(i): i // 3 for i in range(9)}
        names = {key: f'C{8 - value}' for key, value in truth.items()}
        exact = partition_score(names, truth)
        self.assertTrue(exact['exact_recovery'])
        self.assertEqual(exact['ari'], 1.)
        self.assertEqual(exact['null_partition_count'], 280)
        self.assertAlmostEqual(exact['exact_permutation_p'], 1 / 280)
        singleton = partition_score({key: key for key in names}, truth)
        self.assertEqual(singleton['ari'], 0.)
        self.assertEqual(singleton['exact_permutation_p'], 1.)
        self.assertFalse(singleton['exact_recovery'])

    def test_each_identity_gets_only_its_own_maps_and_blind_fields(self):
        data = dict(members=['X', 'Y', 'Z'], map_seeds=list(range(8)),
                    batch={key: torch.arange(24) for key in (*FIELDS, 'reward')})
        group = member_batch(data, 'Y', 2, 3)
        self.assertEqual(set(group), set(FIELDS))
        for values in group.values():
            self.assertEqual(values.tolist(), [7, 10, 13])
