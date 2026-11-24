import unittest

import numpy as np

from scripts.evaluate_lbf_cluster_ego import score_partition


class ClusterEgoEvaluationTests(unittest.TestCase):
    def test_selection_uses_calibration_and_shuffles_keep_repeated_labels(self):
        calibration = np.array([[[1], [1], [0], [0]], [[0], [0], [1], [1]]])
        test = np.repeat(1 - calibration, 4, axis=2)
        result = score_partition(calibration, test, ['a', 'a', 'b', 'b'])
        self.assertEqual(result['cluster_ego_indices'], {'a': 0, 'b': 1})
        self.assertEqual(result['mean_return']['correct_cluster'], 0)
        self.assertAlmostEqual(result['mean_return']['shuffled_cluster'], 2/3)
        self.assertEqual(result['mean_return']['no_cluster'], 0.5)
        self.assertAlmostEqual(result['fraction_shuffled_labels_changed'], 2/3)
        self.assertEqual(result['correct_minus_global']['bootstrap_95_ci'], [-0.5, -0.5])

    def test_one_cluster_has_exactly_the_same_deployed_policy_in_all_arms(self):
        rng = np.random.default_rng(4)
        result = score_partition(rng.random((3, 8, 12)), rng.random((3, 8, 40)), ['C1'] * 8)
        self.assertEqual(result['fraction_shuffled_ego_choices_changed'], 0)
        for key in ('correct_minus_global', 'correct_minus_shuffled'):
            np.testing.assert_allclose(result[key]['bootstrap_95_ci'], [0., 0.], atol=1e-15)
        reference = result['size_matched_partitions']
        self.assertEqual(reference['unique_unlabeled_partitions'], 1)
        self.assertEqual(reference['partitions_tied'], 1)

    def test_size_matched_partitions_refit_only_on_calibration_and_count_each_split_once(self):
        scores = np.array([[[1], [1], [0], [0]], [[0], [0], [1], [1]]])
        result = score_partition(scores, scores, ['a', 'a', 'b', 'b'])['size_matched_partitions']
        self.assertEqual(result['unique_unlabeled_partitions'], 3)
        self.assertEqual((result['partitions_below'], result['partitions_tied'], result['partitions_above']), (2, 1, 0))
        self.assertAlmostEqual(result['mean_random_partition_return'], 2 / 3)
        self.assertAlmostEqual(result['correct_minus_random']['mean_difference'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
