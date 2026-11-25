import math
import unittest

import numpy as np
import torch

from scripts.evaluate_lbf_masked_completion import compare, evaluate_episodes, evaluation_windows


class MaskedCompletionTests(unittest.TestCase):
    def test_each_timestep_is_scored_once_and_hidden_in_a_contiguous_real_span(self):
        lengths = torch.tensor([1, 2, 3, 5])
        filled = torch.arange(5) < lengths[:, None]
        visits = torch.zeros_like(filled, dtype=torch.long)
        for scored, hidden in evaluation_windows(filled, 3):
            visits += scored
            self.assertFalse((scored & ~hidden).any())
            self.assertFalse((hidden & ~filled).any())
            for i in scored.any(1).nonzero().flatten().tolist():
                times = hidden[i].nonzero().flatten().tolist()
                self.assertEqual(times, list(range(times[0], times[-1]+1)))
                self.assertEqual(len(times), max(1, min(3, int(lengths[i])-1)))
        self.assertTrue(torch.equal(visits, filled.long()))

    def test_scoring_uses_actual_slot_and_same_masked_decision(self):
        class RecordingModel(torch.nn.Module):
            n_actions = 6

            def encode(self, states, actions, slots, filled, action_mask=None):
                if action_mask is not None:
                    assert (action_mask & ~slots).sum() == 0
                return action_mask

            def action_log_probs(self, states, actions, slots, filled, context):
                assert context is None
                return -slots.float()

            def action_logits(self, states, context):
                logits = torch.zeros((*context.shape, self.n_actions))
                logits[..., 2] = context.float()*2
                return logits

        batch = dict(filled=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
                     state=torch.zeros(2, 5, 1))
        slots = torch.zeros(2, 4, 2, dtype=torch.bool)
        slots[0, :, 1] = True
        slots[1, :2, 0] = True
        batch['teammate_mask'] = slots
        batch['teammate_actions'] = torch.where(slots, 2, -1)
        choices = torch.zeros(2, 4, 6, dtype=torch.bool)
        choices[..., 2:4] = True
        scores = evaluate_episodes(RecordingModel(), batch, choices, 3)
        den = 5 + math.exp(2)
        for metric, expected in [('full', 1.), ('completion', math.log(den)-2),
                                 ('binary', math.log(den/(math.exp(2)+1)))]:
            values = scores[metric][batch['filled']]
            torch.testing.assert_close(values, torch.full_like(values, expected))

    def test_report_weights_eligible_episodes_and_pairs_entire_maps(self):
        before = torch.tensor([[3.,0.], [1.,3.], [9.,9.], [10.,60.]], dtype=torch.double)
        valid = torch.tensor([[1,0],[1,1],[0,0],[1,0]], dtype=torch.bool)
        sampled_maps = np.array([[0,0],[1,1],[0,1]])
        row = compare(before, torch.zeros_like(before), valid, 2, sampled_maps)
        self.assertEqual((row['episodes'],row['decisions'],row['maps']), (3,4,2))
        self.assertEqual(row['old_mean_nll'], 5.)
        np.testing.assert_allclose(row['paired_95_ci'], np.quantile([2.5,10.,5.],[.025,.975]))
