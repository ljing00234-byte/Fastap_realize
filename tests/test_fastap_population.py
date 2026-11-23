from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from fastap_methods.fastap_population import OnlineTeammates
from fastap_methods.qmix import PROJECT


class OnlineTeammateTests(unittest.TestCase):
    def test_qmix_generated_pair_preserves_both_original_roles(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        temp_root = PROJECT / '.tmp/epymarl'
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as temp:
            source = OnlineTeammates('lbf', temp, seed=3, device='cpu', steps_per_group=25)
            solo = source(0)
            pair = source(1)
            self.assertEqual([len(group.policies) for group in (solo, pair)], [1, 2])
            first, second = [factory() for factory in pair.policies]
            self.assertEqual([first.role.argmax().item(), second.role.argmax().item()], [2, 3])
            self.assertIs(first.model, second.model)
            self.assertFalse(any(p.requires_grad for p in first.model.parameters()))
            self.assertIsNot(first.last_action, second.last_action)
            self.assertEqual(len(set(record['seed'] for record in source.records)), 2)
            self.assertTrue(all(record['training_steps'] >= 25 and record['optimizer_updates'] > 0
                                for record in source.records))
            self.assertEqual([r['group'] for r in source.records], [solo.name, pair.name])
            for record in source.records:
                self.assertTrue((Path(temp) / record['checkpoint'] / 'agent.th').is_file())
            with self.assertRaises(FileExistsError):
                source(0)

    def test_incomplete_or_untrained_group_never_published(self):
        temp_root = PROJECT / '.tmp/epymarl'
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as temp:
            for scene, counts in [('cn', (1, 2)), ('pp', (1,))]:
                source = OnlineTeammates(scene, temp, seed=11, device='cpu', steps_per_group=5)
                self.assertEqual(source.counts, counts)
                with patch('fastap_methods.fastap_population.train_qmix',
                           return_value={'optimizer_updates': 0, 'weight_change': 0.}):
                    with self.assertRaisesRegex(RuntimeError, 'no learned actor update'):
                        source(1)
                self.assertFalse(source.records)
                self.assertEqual(list((Path(temp) / 'teammates').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
