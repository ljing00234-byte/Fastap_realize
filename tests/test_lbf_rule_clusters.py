import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from scripts.evaluate_lbf_crp_prediction import FIELDS, digest
from scripts.validate_lbf_rule_clusters import ROOT, member_batch, partition_score, score, score_partition_on_maps, transfer


class BlindRulePartitionTests(unittest.TestCase):
    def test_loading_rule_pairs_exclude_the_easy_exploration_class(self):
        truth = {str(i): ('BASELINE', 'SOCIAL1', 'EGOISTIC')[i // 3] for i in range(9)}
        labels = {n: 'C1' if r == 'EGOISTIC' else 'C2' for n, r in truth.items()}
        similar = {n: r for n, r in truth.items() if r != 'EGOISTIC'}
        metrics = score_partition_on_maps(labels, truth, similar)
        self.assertGreater(metrics['ari'], 0.)
        self.assertEqual(metrics['similar_rule_pairs'], dict(
            ari=0., families=['BASELINE', 'SOCIAL1'], same_family_joined=6, same_family_total=6,
            different_family_joined=9, different_family_total=9))

    def test_compare_objectives_on_same_blind_maps_and_reveal_key_only_for_scoring(self):
        with TemporaryDirectory(dir=ROOT/'.tmp/epymarl') as temp:
            data_dir = Path(temp)/'reference'
            output_dir = Path(temp)/'experiment'
            data_dir.mkdir()
            names = ['X', 'Y', 'Z', 'W']
            sham = dict.fromkeys(names, 'C1')
            classes = dict(X='A', Y='A', Z='B', W='B')
            for split, maps in [('train', [1, 2]), ('query', [3, 4])]:
                torch.save(dict(members=names, map_seeds=maps), data_dir/f'{split}.pt')
            cfg = dict(order=names, decoder='categorical', seed=7, alpha=.5, lr=.0005,
                       minibatch_size=32, crp_model_steps=64, probe_episodes=2)
            for directory, extra, labels in [(data_dir, {}, sham), (output_dir, {'masked_action_weight': 1.}, classes)]:
                run = directory/'categorical'/'seed7'
                run.mkdir(parents=True)
                record = dict(protocol={**cfg, **extra}, assignments=labels, train_sha256=digest(data_dir/'train.pt'))
                (run/'clustering.json').write_text(json.dumps(record))

            def partitions(checkpoint, query, size):
                self.assertEqual(query['map_seeds'], [3, 4])
                self.assertEqual(size, 2)
                labels = sham if data_dir in checkpoint.parents else classes
                return dict(blocks=[dict(map_seeds=query['map_seeds'], assignments=labels)])

            with patch('scripts.validate_lbf_rule_clusters.frozen_partitions', side_effect=partitions):
                blind = transfer(output_dir, None, [7], data_dir=data_dir, paired_control=data_dir)
            (output_dir/'blind_partitions.json').write_text(json.dumps(blind))
            key = dict(labels=classes, files_sha256={name: digest(data_dir/name) for name in ('train.pt', 'query.pt')})
            (data_dir/'truth.json').write_text(json.dumps(key))
            report = score(output_dir, data_dir)
            self.assertEqual(report['masked_minus_unmasked'], [dict(seed=7, online_ari_delta=1., heldout_ari_deltas=[1.])])

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
