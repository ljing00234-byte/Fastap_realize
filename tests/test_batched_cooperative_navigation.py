from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import yaml

from fastap_methods.batched_qmix import BatchedQMIX
from fastap_methods.cooperative_navigation_eval import check_cn_adapter, validate_cn
from fastap_methods.single_teammate import independent_test
from fastap_scenarios.torch_cooperative_navigation import TorchCooperativeNavigation, check_parity

ROOT = Path(__file__).resolve().parents[1]


class BatchedNavigationTests(unittest.TestCase):
    def test_scalar_parity_including_collision_and_absent_slot(self):
        self.assertTrue(check_parity("cpu")["passed"])
        world = TorchCooperativeNavigation(1, "cpu")
        world.load_snapshot([[[-1,-1], [-1,1], [1,-1]]], torch.zeros(1, 3, 2),
                            [[[-1,-1], [-1,1], [1,-1], [1,1]]])
        reward, _, _ = world.step(torch.zeros(1, 3, dtype=torch.long))
        self.assertAlmostEqual(reward.item(), -2)
        self.assertEqual(world.collisions.item(), 0)
        self.assertTrue((world.state()[:, 15:20] == 0).all())
        self.assertTrue((world.observations()[:, :, -3:] == 0).all())

    def test_train_export_and_replay_through_ego_interface(self):
        cfg = yaml.safe_load((ROOT / "configs/methods/qmix.yaml").read_text())
        cfg.update(batch_size=2, buffer_size=4, bootstrap_at_time_limit=False)
        actor = yaml.safe_load((ROOT / "configs/agents/drqn.yaml").read_text())
        learner = BatchedQMIX(actor, cfg, device="cpu", seed=3, world_type=TorchCooperativeNavigation)
        with TemporaryDirectory(dir=ROOT/".tmp") as tmp:
            run = Path(tmp)
            learner.save_checkpoint(run/"checkpoints/0")
            step = learner.train_batch(TorchCooperativeNavigation(2, "cpu", seed=5), "random")
            self.assertEqual((step["training_steps"], step["optimizer_updates"]), (50, 1))
            chosen = learner.save_checkpoint(run/"checkpoints/50")
            val = validate_cn(learner.actor, learner.spec, n_maps=4, seed=7, device="cpu")
            self.assertTrue(np.isfinite(val["trained"]["mean_return"]))
            report = independent_test(run, chosen, range(4), scene="cn")
            self.assertEqual(report["maps"], 4)
            for score in report["scores"].values():
                self.assertAlmostEqual(score["mean_return"], -25 * score["mean_landmark_distance_sum_per_step"]
                                       - score["mean_pair_collisions_per_episode"])
            replay = check_cn_adapter(run/"T1", [17, 18])
            self.assertEqual(replay["matched_transitions"], 50)
            self.assertEqual((replay["obs_size"], replay["state_size"], replay["actions_per_agent"]), (21, 28, 5))
