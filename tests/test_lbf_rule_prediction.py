import unittest

import numpy as np
import torch

from scripts.evaluate_lbf_crp_prediction import predict
from scripts.evaluate_lbf_rule_prediction import check_splits, predict_by_rule


class RulePredictionTests(unittest.TestCase):
    def test_same_family_substitutions_and_all_other_identities(self):
        def batch(state, values):
            n = len(values)
            return dict(state=torch.full((n, 2, 1), state), filled=torch.ones(n, 1).bool(),
                        teammate_actions=torch.tensor(values).reshape(n, 1, 1),
                        teammate_mask=torch.ones(n, 1, 1).bool())

        class RecordingDecoder(torch.nn.Module):
            def encode(self, states, actions, slots, filled):
                assert (states == 1).all()
                return actions[:, 0].float()

            def episode_action_nll(self, states, actions, slots, filled, context):
                assert (states == 2).all()
                return (actions[:, 0, 0] - context[:, 0]).abs()

        model = RecordingDecoder()
        # Rows alternate the two families, so adjacent identities alternate as well.
        values = [1, 8, 3, 6]
        trials = dict(support=batch(1, values), query=batch(2, values))
        scores = predict_by_rule(model, model, **trials, labels=['x', 'y', 'x', 'y'])
        np.testing.assert_equal(scores['same_rule_other_id'], [[2, 2, 2, 2]])
        np.testing.assert_equal(scores['wrong_rule'], [[6, 6, 4, 4]])
        for key, expected in predict(model, model, trials, 4).items():
            np.testing.assert_allclose(scores[key], expected)

    def test_repeated_initial_states_with_distinct_seeds_are_rejected(self):
        members = ['B01', 'B02']
        batch = dict(state=torch.zeros(2, 2, 1), filled=torch.ones(2, 1).bool(),
                     teammate_actions=torch.zeros(2, 1, 1).long(), teammate_mask=torch.ones(2, 1, 1).bool())
        splits = {name: dict(members=members, map_seeds=[i], batch=batch)
                  for i, name in enumerate(('train', 'support', 'query'))}
        with self.assertRaisesRegex(ValueError, 'repeated map'):
            check_splits(splits, {m: m for m in members})


if __name__ == '__main__':
    unittest.main()
