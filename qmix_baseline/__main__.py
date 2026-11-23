"""Command-line entry point for the project's QMIX baseline."""

import argparse
import os
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
COMMANDS = {
    "train-teammate": ("train_lbf_cooperation.py", "--single-teammate"),
    "generate-pool": ("generate_teammate_pool.py",),
    "train-ego": ("train_pool_ego.py",),
}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="python -m qmix_baseline",
        usage="%(prog)s COMMAND [ARGS ...]",
        description="生成冻结队友并训练 QMIX ego。",
        epilog=(
            "train-teammate  联合训练一支队伍，导出第一名冻结队友 T1\n"
            "generate-pool   从 T1 开始生成队友集合并交叉评估\n"
            "train-ego       使用冻结队友集合训练两个 ego\n\n"
            "运行 %(prog)s COMMAND --help 可查看该命令的参数。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND")
    command = parser.parse_args(args[:1]).command
    filename, *defaults = COMMANDS[command]
    # Keep the existing script path visible to the GPU monitor and worker processes.
    os.execv(sys.executable, [sys.executable, str(SCRIPTS / filename), *defaults, *args[1:]])


if __name__ == "__main__":
    main()
