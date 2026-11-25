from contextlib import closing
import hashlib
import unittest

import numpy as np

from fastap_methods.lbf_probe_starts import pre_load_moves, select_probe_starts
from fastap_scenarios.fastap_rollout import FastapRolloutEnv
from fastap_scenarios.policies import LBF_DEMO, RANDOM_SOLO


class LbfProbeStartTests(unittest.TestCase):
    def test_preload_requires_visible_food_and_both_legal_cardinal_choices(self):
        obs = np.array([[0, 0, 5], [-1, -1, 0], [-1, -1, 0], [1, 1, 2],
                        [-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]).ravel()
        available = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
        np.testing.assert_equal(pre_load_moves(obs, available), [True, False, False, False])
        # With a second food to the southeast, every legal move approaches a nearest food.
        obs[3:6] = [2, 2, 5]
        self.assertFalse(pre_load_moves(obs, available).any())
        obs[3:6] = [-1, -1, 0]
        available[5] = True
        self.assertFalse(pre_load_moves(obs, available).any())
        available[5] = False
        available[[2, 4]] = False
        self.assertFalse(pre_load_moves(obs, available).any())
        available[[2, 4]] = True
        obs[:3] = [-1, -1, 0]
        self.assertFalse(pre_load_moves(obs, available).any())

    def test_replay_matches_first_eligible_starts_in_the_real_collector(self):
        selected = select_probe_starts(834, 4)
        self.assertEqual(selected, select_probe_starts(834, 4))
        actual = []
        for seed in range(834, 834 + selected['candidate_count']):
            with closing(FastapRolloutEnv('lbf', [RANDOM_SOLO], seed=seed)) as env:
                _, state, _ = env.reset(0)
                obs = env.world.observations()[2]
                available = env.world.avail_actions()[2]
                if pre_load_moves(obs, available).any():
                    actual.append((seed, hashlib.sha256(state.tobytes()).hexdigest()))
        self.assertEqual(actual, list(zip(selected['map_seeds'], selected['initial_state_sha256'])))
        self.assertEqual(len(set(selected['initial_state_sha256'])), 4)
        control = select_probe_starts(834, 4, mode='shared_random')
        self.assertEqual(control['map_seeds'], list(range(834, 838)))
        # Limiting the search must fail explicitly, rather than admitting an unqualified board.
        with self.assertRaisesRegex(ValueError, 'distinct shared_preload'):
            select_probe_starts(834, 4, max_candidates=4)

    def test_four_player_boards_offer_a_choice_to_both_teammates(self):
        selected = select_probe_starts(834, 3, n_teammates=2)
        self.assertEqual(selected['target_roles'], [2, 3])
        self.assertEqual(selected['n_teammates'], 2)
        for seed, fingerprint in zip(selected['map_seeds'], selected['initial_state_sha256']):
            with closing(FastapRolloutEnv('lbf', [LBF_DEMO[1]], seed=seed)) as env:
                _, state, _ = env.reset(0)
                self.assertEqual(hashlib.sha256(state.tobytes()).hexdigest(), fingerprint)
                obs, legal = env.world.observations(), env.world.avail_actions()
                for role in (2, 3):
                    self.assertTrue(pre_load_moves(obs[role], legal[role]).any())
