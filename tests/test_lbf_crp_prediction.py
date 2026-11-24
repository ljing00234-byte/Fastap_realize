import unittest

import numpy as np
import torch

from scripts.evaluate_lbf_crp_prediction import predict, summarize


class CrossMapPredictionTests(unittest.TestCase):
    def test_support_only_encoding_all_wrong_identities_and_common_query(self):
        # Each map has three members. Context records which member was encoded.
        # Mark B with a different state value so accidental query encoding fails.
        n = 6
        def batch(marker):
            return dict(state=torch.full((n, 2, 1), marker), filled=torch.ones(n, 1).bool(),
                        teammate_actions=torch.tensor([1, 2, 3, 4, 5, 6]).reshape(n, 1, 1),
                        teammate_mask=torch.ones(n, 1, 1).bool())

        class RecordingDecoder(torch.nn.Module):
            def encode(self, states, actions, slots, filled):
                assert torch.all(states == 17)
                return actions[:, 0].float()

            def episode_action_nll(self, states, actions, slots, filled, context):
                assert torch.all(states == 43)
                return (actions[:, 0, 0] - context[:, 0]).abs()

        result = predict(RecordingDecoder(), RecordingDecoder(), dict(support=batch(17), query=batch(43)), 3)
        np.testing.assert_array_equal(result["correct"], [[0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(result["wrong"], [[1.5, 1, 1.5], [1.5, 1, 1.5]])
        np.testing.assert_array_equal(result["state_only"], [[1, 2, 3], [4, 5, 6]])

    def test_interval_resamples_whole_map_and_gate_requires_both_controls(self):
        scores = dict(correct=np.array([[1., 2.], [1., 2.]]), wrong=np.array([[3., 4.], [.5, 1.5]]),
                      state_only=np.array([[2., 3.], [2., 3.]]))
        rows = np.array([[0, 0], [1, 1]])
        result = summarize(scores, rows, [11, 22], ["T1", "T2"])
        self.assertFalse(result["both_lower_bounds_positive"])
        self.assertEqual(result["gain"]["wrong"]["mean_nats"], .75)
        self.assertEqual(result["gain"]["state_only"]["paired_95_ci"], [1., 1.])


if __name__ == "__main__":
    unittest.main()
