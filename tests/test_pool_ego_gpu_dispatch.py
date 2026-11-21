"""GPU held-out recovery must use the PP run's recorded device and shared monitor."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_pool_ego as entry


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
