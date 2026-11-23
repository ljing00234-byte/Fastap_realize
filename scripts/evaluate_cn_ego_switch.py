"""Test a mixed-trained CN ego against changes of teammate policy and roster size."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from gpu2_monitor import EXPECTED_PEAK, OUTPUTS, ROOT, require_monitored_cuda, run_monitored, write_json
from run_baseline import pool_paths, project_path


def evaluate(args):
    import torch
    from fastap_agents import load_frozen_agent
    from fastap_methods.cn_ego_switch_eval import evaluate_ego_switch
    from fastap_methods.cn_pool_ego_eval import BOARD_GENERATOR, make_evaluation_boards
    from fastap_methods.particle_eval import serialize_scores

    uuid = require_monitored_cuda()
    device = "cuda:0"
    torch.set_num_threads(1)
    config, protocol, status, heldout = (
        json.loads((args.ego_run / f"{name}.json").read_text())
        for name in ("config", "protocol", "status", "heldout")
    )
    members = protocol["pool_members"]
    names = [m["id"] for m in members]
    checkpoint = (args.ego_run / status["final_checkpoint"]).resolve()
    if ((config["scene"], config["control"], config["method"]["name"]) != ("cn", "ego", "qmix")
            or config.get("training_teammate") is not None or status["phase"] != "complete"
            or protocol["training_roster_sizes"] != [1] or protocol["training_pool_order"] != names
            or len(set(names)) != len(names) or len(names) < 2
            or any(status["sampling_counts"][name] <= 0 for name in names)
            or checkpoint.parent != args.ego_run / "checkpoints" or checkpoint.name != str(status["training_steps"])
            or heldout["primary_checkpoint"] != status["final_checkpoint"]):
        raise ValueError("Expected a completed mixed-partner CN QMIX run and its final held-out checkpoint")
    pool = project_path(config["teammate_pool"])
    if OUTPUTS not in pool.parents:
        raise ValueError("Expected the published CN pool under project outputs")
    paths = pool_paths(pool, "cn")
    published = json.loads(pool.read_text())["members"]
    if [dict(id=m["id"], sha256=m["sha256"]) for m in published] != members:
        raise ValueError("Published teammate IDs or weights differ from this training run")

    first_seed, maps = protocol["heldout_first_seed"], protocol["heldout_map_seeds_per_roster"]
    static = heldout["scores"]["final"]
    if (heldout["first_seed"] != first_seed or static["first_map_seed"] != first_seed
            or static["n_maps_per_teammate"] != maps or static["n_episodes"] != maps * len(names)
            or static["board_generator"] != BOARD_GENERATOR or static["board_device"] != device
            or static["pool_order"] != names):
        raise ValueError("Stored stationary scores use different maps, roster or board generator")
    static["episode_returns"] = torch.tensor([static["episode_returns"][name] for name in names],
                                            dtype=torch.float64, device=device)
    if (static["episode_returns"].shape != (len(names), maps)
            or abs(static["episode_returns"].mean().item() - static["mean_return"]) > 1e-9):
        raise ValueError("Stored stationary map-by-partner scores disagree with aggregate")
    ego = load_frozen_agent(checkpoint, device=device)
    teammates = {name: load_frozen_agent(path, device=device) for name, path in zip(names, paths)}
    boards = make_evaluation_boards(first_seed, maps, device=device)
    write_json(args.output_dir / "protocol.json", dict(
        scene="cn", method="vanilla QMIX", training_run=str(args.ego_run.relative_to(ROOT)),
        training_seed=config["seed"], training_steps=status["training_steps"],
        checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=hashlib.sha256((checkpoint / "agent.th").read_bytes()).hexdigest(),
        training_roster_sizes=protocol["training_roster_sizes"], sampling_counts=status["sampling_counts"],
        pool_members=members, board_generator=BOARD_GENERATOR, first_map_seed=first_seed,
        maps_per_initial_teammate=maps, horizon=protocol["episode_limit"], schedule_seed=args.schedule_seed,
        interval=list(args.interval), device=device, physical_gpu_index=2, gpu_uuid=uuid,
        stationary_source=str((args.ego_run / "heldout.json").relative_to(ROOT)),
        scoring=protocol["scoring"],
        design="Start with one teammate on the original held-out maps. Reuse the stationary final scores. "
               "In both new conditions, redraw the first teammate on the same per-map schedule. "
               "The open-roster condition also redraws one or two teammates with equal probability; "
               "two members of a group have distinct IDs. The egos keep their history.",
        uncertainty="Bootstrap whole maps with all starting teammate IDs kept together.",
        source="https://proceedings.mlr.press/v216/zhang23a/zhang23a-supp.pdf#page=7"))
    report = evaluate_ego_switch(ego, teammates, boards, schedule_seed=args.schedule_seed,
                                 interval=tuple(args.interval), stationary=static)
    for mode in ("stationary", "policy_switch", "open_switch"):
        report[mode] = serialize_scores(report[mode])
    write_json(args.output_dir / "comparison.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ego_run", type=project_path)
    parser.add_argument("output_dir", type=project_path)
    parser.add_argument("--interval", nargs=2, type=int, default=(5, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--schedule-seed", type=int, default=5_400_000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.ego_run == args.output_dir or any(OUTPUTS not in p.parents for p in (args.ego_run, args.output_dir)):
        parser.error("Use distinct project output directories")
    if args.schedule_seed < 0 or not 1 <= args.interval[0] <= args.interval[1]:
        parser.error("Expected a nonnegative seed and ordered positive waiting-time bounds")
    if not args.worker:
        for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
            if ROOT not in Path(os.environ.get(key, "/")).resolve().parents:
                parser.error("Source scripts/epymarl-env.sh before launching")
        run_monitored(args.output_dir, [str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], EXPECTED_PEAK)
        return
    sys.path.insert(0, str(ROOT))
    write_json(args.output_dir / "status.json", {"phase": "evaluation"})
    try:
        evaluate(args)
    except BaseException as exc:
        write_json(args.output_dir / "status.json", {"phase": "failed", "error": repr(exc)})
        raise
    write_json(args.output_dir / "status.json", {"phase": "complete"})


if __name__ == "__main__":
    main()
