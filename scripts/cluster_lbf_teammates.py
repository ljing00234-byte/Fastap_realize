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
    parser.add_argument("--seed", type=int, default=7, help="behavior-model seed (probe seed is one higher)")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("The seed must be nonnegative")
    if os.environ.get("FASTAP_ROOT") != str(ROOT):
        parser.error("Source scripts/epymarl-env.sh for project-local caches and temporary files")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    import torch
    from fastap_methods.lbf_crp_analysis import run_crp

    torch.set_num_threads(1)
    result = run_crp(args.teammate_pool, args.run_dir, seed=args.seed)
    print(json.dumps(result["clusters"], indent=2))


if __name__ == "__main__":
    main()
