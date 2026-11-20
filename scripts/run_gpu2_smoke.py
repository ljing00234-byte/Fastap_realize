"""Run the original QMIX/LBF smoke experiment through the shared GPU 2 monitor."""

import argparse
from pathlib import Path

from gpu2_monitor import EXPECTED_PEAK, ROOT, run_monitored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="fresh directory inside project outputs/")
    parser.add_argument("--scene", choices=("upstream", "fastap"), default="upstream")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    worker_args = [str(ROOT / "scripts/qmix_lbf_smoke.py"), str(run_dir), "--scene", args.scene]
    run_monitored(run_dir, worker_args, EXPECTED_PEAK)


if __name__ == "__main__":
    main()
