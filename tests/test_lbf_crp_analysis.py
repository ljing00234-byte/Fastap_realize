import unittest

import numpy as np

from scripts.plot_lbf_crp import aggregate_crossplay, project_embeddings


class LbfCrpAnalysisTests(unittest.TestCase):
    def test_directed_blocks_and_within_cluster_swaps(self):
        # The two C1 members are nonadjacent in the original matrix.
        matrix = [[1., .1, .2], [.3, 1., .9], [.6, .7, 1.]]
        crossplay = dict(ids=['T1', 'T2', 'T3'], n_maps=256, first_map_seed=870000,
                         crossplay=[[dict(mean_return=x) for x in row] for row in matrix])
        assignments = dict(T1='C1', T2='C2', T3='C1')
        result = aggregate_crossplay(crossplay, assignments)
        np.testing.assert_allclose(result['mean_return'], [[.7, .4], [.6, 1.]])
        self.assertEqual(result['pairings'], [[4, 2], [2, 1]])
        swaps = result['summaries']['within_cluster_other_members']
        self.assertEqual(swaps['pairings'], 2)
        self.assertAlmostEqual(swaps['mean_return'], .4)
        for assignments, empty in [
                (dict(T1='C1', T2='C2', T3='C3'), 'within_cluster_other_members'),
                (dict(T1='C1', T2='C1', T3='C1'), 'across_clusters')]:
            summary = aggregate_crossplay(crossplay, assignments)['summaries']
            self.assertEqual(summary[empty], dict(pairings=0, mean_return=None))

    def test_common_pca_preserves_trajectory_labels(self):
        rows = [
            dict(teammate=t, cluster=c, episode=i, embedding=x)
            for t, c, i, x in [('T1', 'C1', 1, [2, 1, 4]), ('T2', 'C2', 1, [5, 3, 4]),
                               ('T1', 'C1', 2, [3, 2, 5]), ('T2', 'C2', 2, [4, 4, 1])]
        ]
        result = project_embeddings(rows, dict(T1='C1', T2='C2'))
        coords = np.array([row['xyz'] for row in result['rows']])
        original = np.array([row['embedding'] for row in rows])
        np.testing.assert_allclose(coords.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(coords @ result['components'] + result['center'], original)
        self.assertAlmostEqual(sum(result['explained_variance_ratio']), 1.)
        self.assertEqual([(r['teammate'], r['episode']) for r in result['rows']],
                         [(r['teammate'], r['episode']) for r in rows])


if __name__ == '__main__':
    unittest.main()
