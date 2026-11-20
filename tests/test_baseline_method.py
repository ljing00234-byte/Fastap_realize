import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from fastap_methods import train_qmix


PROJECT = Path(__file__).resolve().parents[1]


class BaselineMethodTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        temp_root = PROJECT / ".tmp" / "epymarl"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp = TemporaryDirectory(dir=temp_root)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def train(self, name, **overrides):
        settings = dict(t_max=6, test_interval=6, batch_size=2, buffer_size=4,
                        test_nepisode=1, save_model_interval=6, learner_log_interval=1)
        settings.update(overrides)
        return train_qmix(
            scene_config={"env": "fastap_scenario", "runner": "fastap_episode",
                          "env_args": {"key": "cn", "control": "team", "n_teammates": 2,
                                       "world_options": {"episode_limit": 2}}},
            agent_config={"hidden_dim": 8, "obs_last_action": True, "use_rnn": True},
            method_config=settings, run_dir=self.root / name, seed=29, device="cpu",
        )

    def test_evaluation_frequency_does_not_change_learned_policy(self):
        final_only = self.train("final")
        frequent = self.train("frequent", test_interval=2)
        self.assertEqual(final_only["optimizer_updates"], 2)
        self.assertGreater(final_only["weight_change"], 0)
        self.assertEqual(final_only["final_td_loss"], frequent["final_td_loss"])
        for component in ("agent.th", "mixer.th"):
            first = torch.load(self.root / "final" / final_only["checkpoint"] / component, weights_only=True)
            second = torch.load(self.root / "frequent" / frequent["checkpoint"] / component, weights_only=True)
            for key in first:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        with self.assertRaises(FileExistsError):
            self.train("final")

    def test_replay_keeps_raw_reward_and_true_terminal_vs_time_limit(self):
        from fastap_scenarios.cooperative_navigation import CooperativeNavigationWorld
        original = CooperativeNavigationWorld.step

        def one_step_game(world, joint_actions):
            _, _, info = original(world, joint_actions)
            # Two equally long episodes so a timeout and a terminal coexist in one sample.
            world.test_visits = getattr(world, "test_visits", 0) + 1
            info["episode_limit"] = (world.test_visits == 2)
            return (-10.0 if info["episode_limit"] else 20.0), True, info

        samples = []
        # Import the same learner class that the training registry constructs.
        sys.path.insert(0, str(PROJECT / "third_party" / "epymarl" / "src"))
        self.addCleanup(sys.path.pop, 0)
        from learners.q_learner import QLearner
        update = QLearner.train

        def record_then_update(learner, batch, *args):
            samples.extend(zip(batch["reward"][:, 0, 0].tolist(), batch["terminated"][:, 0, 0].tolist()))
            self.assertFalse(learner.args.standardise_rewards)
            return update(learner, batch, *args)

        with patch.object(CooperativeNavigationWorld, "step", one_step_game), \
                patch.object(QLearner, "train", record_then_update):
            result = self.train("raw", t_max=2)
        self.assertCountEqual(samples, [(20, 1), (-10, 0)])
        self.assertEqual(result["eval_return_mean"], 20)
        self.assertEqual(json.loads((self.root / "raw/summary.json").read_text()), result)
