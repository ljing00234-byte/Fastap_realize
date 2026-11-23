"""Check the training and evaluation dispatch of the pool ego entry points."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_pool_ego as entry
import train_lbf_cooperation as team_entry


class TimeLimitDispatchTests(unittest.TestCase):
    def test_cn_ego_uses_published_single_teammate_roster_and_finite_horizon(self):
        run_dir = str(ROOT / "outputs" / "mock-cn-run")
        argv = ["pool_ego", run_dir, "--scene", "cn", "--worker"]
        with patch.object(sys, "argv", argv), patch.object(entry, "train") as train, \
             patch.object(entry, "require_monitored_cuda", return_value="GPU-test"):
            entry.main()
        args = train.call_args.args[0]
        self.assertIs(args.bootstrap_at_time_limit, False)
        self.assertEqual(args.teammate_pool, ROOT / "outputs/cn_teammate_pool_01/pool.json")
        self.assertEqual(entry.SCENE_PROTOCOLS["cn"]["training_roster_sizes"], [1])
        world, evaluation = entry.scene_modules("cn")
        self.assertEqual((world.scene_key, world.n_agents, world.episode_limit), ("cn", 2, 25))
        self.assertIs(evaluation.TorchCNEgo, world)

    def test_workers_resolve_and_record_default_and_explicit_timeout_modes(self):
        run_dir = str(ROOT / "outputs" / "mock-time-limit-run")
        for scene, flags, expected in (("lbf", [], False), ("lbf", ["--bootstrap-at-time-limit"], True),
                                       ("pp", [], True), ("pp", ["--no-bootstrap-at-time-limit"], False)):
            with self.subTest(scene=scene, flags=flags):
                argv = ["pool_ego", run_dir, "--worker", "--scene", scene, *flags]
                with patch.object(sys, "argv", argv), patch.object(entry, "train") as train, \
                     patch.object(entry, "require_monitored_cuda", return_value="GPU-test"):
                    entry.main()
                self.assertIs(train.call_args.args[0].bootstrap_at_time_limit, expected)

                with patch.object(sys, "argv", [*argv, "--single-teammate"]), \
                     patch.object(team_entry, "require_monitored_cuda", return_value="GPU-test"), \
                     patch.object(team_entry, "write_json") as write, \
                     patch("fastap_methods.single_teammate.run_single_team") as train_team:
                    team_entry.main()
                self.assertIs(write.call_args.args[1]["method"]["bootstrap_at_time_limit"], expected)
                self.assertIs(train_team.call_args.args[2]["bootstrap_at_time_limit"], expected)


class PPTrainingTeammateTests(unittest.TestCase):
    def test_fixed_training_partner_cli_applies_only_to_pp(self):
        argv = ["pool_ego", str(ROOT / "outputs" / "mock-partner-run"), "--worker", "--training-teammate", "T1"]
        with patch.object(entry, "train") as train, patch.object(entry, "require_monitored_cuda", return_value="GPU-test"):
            with patch.object(sys, "argv", [*argv, "--scene", "pp"]):
                entry.main()
            self.assertEqual(train.call_args.args[0].training_teammate, "T1")
            with patch.object(sys, "argv", [*argv, "--scene", "lbf"]), patch.object(sys, "stderr"), \
                 self.assertRaises(SystemExit):
                entry.main()
            self.assertEqual(train.call_count, 1)

    def test_select_training_teammates_keeps_evaluation_pool_intact(self):
        teammates = {"T2": object(), "T1": object()}
        self.assertEqual(list(entry.select_training_teammates(teammates, None)), ["T2", "T1"])
        self.assertEqual(entry.select_training_teammates(teammates, "T1"), {"T1": teammates["T1"]})
        with self.assertRaises(ValueError):
            entry.select_training_teammates(teammates, "T9")
        self.assertEqual(list(teammates), ["T2", "T1"])

    def test_only_training_rollouts_use_the_selected_teammate(self):
        # Keep T1 away from index zero so that mislabeled sampling counts are detectable.
        names = ["T2", "T1", "T3", "T4", "T5", "T6", "T7", "T8"]
        members = [{"id": name, "sha256": name} for name in names]
        world_type = Mock(episode_limit=25)
        world_type.return_value.sample_counts.cpu.return_value.tolist.return_value = [1]
        world_type.return_value.roster_counts.cpu.return_value.tolist.return_value = [1]
        evaluation = Mock(BOARD_GENERATOR="test-boards", BOOTSTRAP_SEED=1, BOOTSTRAP_RESAMPLES=10)
        evaluation.evaluate_pool_ego.return_value = {"mean_return": 1.0}
        learner = Mock(steps=0, episodes=1, updates=0)
        learner.save_checkpoint.side_effect = lambda path: path

        def finish_rollout(world, layout):
            learner.steps = 25
            return {"training_steps": learner.steps}

        learner.train_batch.side_effect = finish_rollout
        heldout = {"scores": {k: {"mean_return": 1.0} for k in ("initial", "final", "validation_selected")}}
        with TemporaryDirectory(dir=ROOT / ".tmp") as tmp:
            directory = Path(tmp)
            pool = directory / "pool.json"
            pool.write_text(json.dumps({"members": members, "target_size": len(members)}))
            paths = [directory / name for name in names]
            args = SimpleNamespace(scene="pp", num_envs=1, seed=1107, total_steps=25,
                                   training_teammate="T1", bootstrap_at_time_limit=False,
                                   run_dir=directory, teammate_pool=pool,
                                   validation_seed=100, validation_maps=1, heldout_seed=200, heldout_maps=1)
            with patch.object(entry, "pool_paths", return_value=paths), \
                 patch("fastap_agents.load_frozen_agent", side_effect=lambda path, device: path.name), \
                 patch("fastap_methods.batched_qmix.BatchedQMIX", return_value=learner), \
                 patch.object(entry, "scene_modules", return_value=(world_type, evaluation)), \
                 patch.object(entry, "check_particle_devices", return_value={}), \
                 patch.object(entry, "evaluate_final_models", return_value=heldout) as final:
                entry.train(args, device="cuda:0", gpu_uuid="GPU-test")
            self.assertEqual(world_type.call_args.kwargs["teammates"], ["T1"])
            self.assertEqual(list(evaluation.evaluate_pool_ego.call_args.args[2]), names)
            self.assertEqual(final.call_args.args[2:4], (members, paths))
            protocol = json.loads((directory / "protocol.json").read_text())
            self.assertEqual(protocol["training_pool_order"], ["T1"])
            self.assertEqual(protocol["heldout_episodes_per_policy"], len(names))
            counts = json.loads((directory / "status.json").read_text())["sampling_counts"]
            self.assertEqual(counts["T1"], 1)
            self.assertEqual(sum(counts.values()), 1)


class PoolEgoRecoveryDispatchTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run = self.root / "outputs" / "pp-run"
        self.run.mkdir(parents=True)
        for name, data in {
            "config": {"scene": "pp", "gpu_uuid": "GPU-test"},
            "protocol": {"evaluation_device": "cuda:0"},
            "status": {"phase": "heldout_evaluation"},
            "monitor": {"finished_at": "done"},
        }.items():
            (self.run / f"{name}.json").write_text(json.dumps(data))

    def test_pp_recovery_launches_a_separate_monitored_attempt(self):
        env = {k: str(self.root / ".cache") for k in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH")}
        env["FASTAP_ROOT"] = str(self.root)
        with patch.object(entry, "ROOT", self.root), patch.dict(os.environ, env), \
             patch.object(sys, "argv", ["train_pool_ego", str(self.run), "--finish-heldout"]), \
             patch.object(entry, "run_monitored") as launch, patch.object(entry, "recover_heldout") as recovery:
            entry.main()
        attempt, worker, peak = launch.call_args.args
        self.assertEqual(attempt.parent, self.run)
        self.assertEqual(worker, [entry.__file__, str(self.run), "--finish-heldout", "--worker"])
        self.assertEqual(peak, entry.BASELINE_PEAK)
        recovery.assert_not_called()

    def test_monitored_worker_uses_recorded_pp_gpu(self):
        with patch.object(entry, "ROOT", self.root), \
             patch.object(sys, "argv", ["train_pool_ego", str(self.run), "--finish-heldout", "--worker"]), \
             patch.object(entry, "require_monitored_cuda", return_value="GPU-test") as binding, \
             patch.object(entry, "recover_heldout") as recovery:
            entry.main()
        binding.assert_called_once_with()
        recovery.assert_called_once_with(self.run, device="cuda:0")
