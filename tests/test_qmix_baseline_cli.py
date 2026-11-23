import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from qmix_baseline.__main__ import main


class QMIXBaselineCLITest(unittest.TestCase):
    def test_commands_delegate_to_the_original_scripts(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        cases = [
            ("train-teammate", "train_lbf_cooperation.py", ["--single-teammate"]),
            ("generate-pool", "generate_teammate_pool.py", []),
            ("train-ego", "train_pool_ego.py", []),
        ]
        for command, filename, defaults in cases:
            with self.subTest(command=command), patch("qmix_baseline.__main__.os.execv") as launch:
                path = scripts / filename
                self.assertTrue(path.is_file())
                args = ["outputs/a trial", "--scene", "cn", "--help"]
                main([command, *args])
                launch.assert_called_once_with(
                    sys.executable, [sys.executable, str(path), *defaults, *args]
                )
