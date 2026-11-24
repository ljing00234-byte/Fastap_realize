"""Fit Fastap's CRP on trajectories from the eight frozen single-teammate LBF policies."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FORMAL_POOL = ROOT / "outputs/fastap_lbf_formal_pool_01/pool.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="new project directory for assignments, probes and embeddings")
    parser.add_argument("--teammate-pool", type=Path, default=FORMAL_POOL, help="eight-member LBF pool.json")
    parser.add_argument("--seed", type=int, default=7, help="behavior-model seed")
    parser.add_argument("--probe-seed", type=int, help="fix probe maps and partners' RNG across fits (default seed + 1)")
    parser.add_argument("--decoder", choices=("paper", "categorical"), default="paper")
    parser.add_argument("--probe-ego", choices=("own_original", "random_legal"), default="own_original",
                        help="use each teammate's original QMIX partners (default) or uniformly random legal actions")
    args = parser.parse_args()
    if args.seed < 0 or (args.probe_seed is not None and args.probe_seed < 0):
        parser.error("Seeds must be nonnegative")
    if os.environ.get("FASTAP_ROOT") != str(ROOT):
        parser.error("Source scripts/epymarl-env.sh for project-local caches and temporary files")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    import torch
    from fastap_methods.lbf_crp_analysis import run_crp

    torch.set_num_threads(1)
    result = run_crp(args.teammate_pool, args.run_dir, seed=args.seed, probe_ego=args.probe_ego,
                     probe_seed=args.probe_seed, decoder=args.decoder)
    print(json.dumps(result["clusters"], indent=2))


if __name__ == "__main__":
    main()
