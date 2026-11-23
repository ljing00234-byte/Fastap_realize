"""Evaluate a trained LBF ego on paired stationary and changing teammate groups."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from gpu2_monitor import OUTPUTS, ROOT, require_monitored_cuda, run_monitored, write_json
from run_baseline import pool_paths, project_path


def run_evaluation(args):
    import torch
    from fastap_agents import load_frozen_agent
    from fastap_methods.lbf_ego_switch_eval import evaluate_ego_switch

    uuid = require_monitored_cuda()
    device = "cuda:0"
    config, protocol, status = [
        json.loads((args.ego_run / name).read_text())
        for name in ("config.json", "protocol.json", "status.json")
    ]
    if ((config["scene"], config["control"], config["method"]["name"]) != ("lbf", "ego", "qmix")
            or status["phase"] != "complete" or protocol["episode_limit"] != 25):
        raise ValueError("Expected a completed LBF QMIX ego run with a 25-step horizon")
    checkpoint = (args.ego_run / status["final_checkpoint"]).resolve()
    if checkpoint.parent != args.ego_run / "checkpoints" or checkpoint.name != str(status["training_steps"]):
        raise ValueError("Invalid final checkpoint in the completed run")
    pool = project_path(config["teammate_pool"])
    if OUTPUTS not in pool.parents:
        raise ValueError("Expected a published project-local teammate pool")
    members = json.loads(pool.read_text())["members"]
    paths = pool_paths(pool, "lbf")
    if [dict(id=m["id"], sha256=m["sha256"]) for m in members] != protocol["pool_members"]:
        raise ValueError("Published teammate pool differs from the pool used to train this ego")
    ego = load_frozen_agent(checkpoint, device=device)
    teammates = {m["id"]: load_frozen_agent(path, device=device) for m, path in zip(members, paths)}
    if len(teammates) != len(members):
        raise ValueError("Duplicate ID in the published teammate pool")
    metadata = dict(
        scene="lbf", method="vanilla QMIX", training_seed=config["seed"],
        training_steps=status["training_steps"], training_run=str(args.ego_run.relative_to(ROOT)),
        bootstrap_at_time_limit=config["method"]["bootstrap_at_time_limit"],
        checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256((checkpoint / "agent.th").read_bytes()).hexdigest(),
        teammate_pool=str(pool.relative_to(ROOT)), pool_members=protocol["pool_members"],
        first_map_seed=args.first_seed, maps_per_starting_roster=args.maps, schedule_seed=args.schedule_seed,
        interval=list(args.interval), horizon=protocol["episode_limit"], chunk_size=args.chunk_size,
        training_teammate_sampling=protocol["teammate_sampling"],
        test_protocol="Paired greedy evaluation with one or two frozen teammates from the training pool. "
                      "Keep ego memory. Redraw waiting times and teammate groups before the next decision. "
                      "Waiting times are discrete uniform, inclusive. Redrawing the same group is allowed.",
        source="https://proceedings.mlr.press/v216/zhang23a/zhang23a-supp.pdf#page=7",
        device=device, physical_gpu_index=2, gpu_uuid=uuid)
    write_json(args.output_dir / "protocol.json", metadata)
    torch.set_num_threads(1)
    report = evaluate_ego_switch(ego, teammates, first_seed=args.first_seed, n_maps=args.maps,
                                 schedule_seed=args.schedule_seed, interval=tuple(args.interval),
                                 chunk_size=args.chunk_size)
    write_json(args.output_dir / "comparison.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ego_run", type=project_path)
    parser.add_argument("output_dir", type=project_path)
    parser.add_argument("--maps", type=int, default=1024, help="Maps per starting roster size and initial partner")
    parser.add_argument("--first-seed", type=int, default=5_100_000)
    parser.add_argument("--schedule-seed", type=int, default=5_200_000)
    parser.add_argument("--interval", nargs=2, type=int, default=(5, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--expected-peak-mib", type=int, default=2048)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (OUTPUTS not in args.ego_run.parents or OUTPUTS not in args.output_dir.parents
            or args.ego_run == args.output_dir):
        parser.error("Choose separate project output directories for the source run and the evaluation")
    if (min(args.maps, args.chunk_size, args.expected_peak_mib) < 1 or min(args.first_seed, args.schedule_seed) < 0
            or not 1 <= args.interval[0] <= args.interval[1]):
        parser.error("Expected positive map counts, chunk size, memory estimate and waiting times; nonnegative seeds")
    if not args.worker:
        for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
            if ROOT not in Path(os.environ.get(key, "/")).resolve().parents:
                parser.error("Source scripts/epymarl-env.sh before launching")
        print(f"Output {args.output_dir}; disk free {shutil.disk_usage(ROOT).free // (1024 ** 3)} GiB", flush=True)
        run_monitored(args.output_dir, [str(Path(__file__).resolve()), *sys.argv[1:], "--worker",
                                        "--expected-peak-mib", str(args.expected_peak_mib)],
                      args.expected_peak_mib)
        return
    sys.path.insert(0, str(ROOT))
    write_json(args.output_dir / "status.json", {"phase": "evaluation"})
    try:
        run_evaluation(args)
    except BaseException as exc:
        write_json(args.output_dir / "status.json", {"phase": "failed", "error": repr(exc)})
        raise
    write_json(args.output_dir / "status.json", {"phase": "complete"})


if __name__ == "__main__":
    main()
