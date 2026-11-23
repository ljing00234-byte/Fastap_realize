"""Evaluate the mixed and T1-trained PP egos against identical within-episode teammate redraws."""

import argparse
import hashlib
import os
from pathlib import Path
import sys

from compare_pp_ego_training_partners import DEVICE, PARTNERS, read_run
from gpu2_monitor import EXPECTED_PEAK, OUTPUTS, ROOT, require_monitored_cuda, run_monitored, write_json
from run_baseline import pool_paths, project_path


def evaluate(args):
    import torch
    from fastap_agents import load_frozen_agent
    from fastap_methods.particle_eval import compare_ego_returns, serialize_scores
    from fastap_methods.pp_ego_switch_eval import evaluate_ego_switch, make_switch_schedule
    from fastap_methods.pp_pool_ego_eval import make_evaluation_boards

    uuid = require_monitored_cuda()
    torch.set_num_threads(1)
    mixed = read_run(args.mixed_run, PARTNERS, uuid)
    fixed = read_run(args.fixed_run, ["T1"], uuid)
    if mixed[:2] != fixed[:2] or mixed[2]["optimizer_updates"] != fixed[2]["optimizer_updates"]:
        raise ValueError("Expected matched training and evaluation settings for the two PP models")
    config, training_protocol = mixed[:2]
    pool = project_path(config["teammate_pool"])
    if OUTPUTS not in pool.parents:
        raise ValueError("Expected the published PP teammate pool under project outputs")
    paths = pool_paths(pool, "pp")
    if (len(paths) != len(PARTNERS) or
            [hashlib.sha256((p / "agent.th").read_bytes()).hexdigest() for p in paths] !=
            [m["sha256"] for m in training_protocol["pool_members"]]):
        raise ValueError("Teammate weights differ from the training runs")
    teammates = {n: load_frozen_agent(p, device=DEVICE) for n, p in zip(PARTNERS, paths)}
    first_seed = training_protocol["heldout_first_seed"]
    n_maps = training_protocol["heldout_map_seeds_per_roster"]
    boards = make_evaluation_boards(first_seed, n_maps, device=DEVICE)
    schedule = make_switch_schedule(boards, PARTNERS, schedule_seed=args.schedule_seed, interval=tuple(args.interval))
    reports, checkpoints = {}, {}
    for name, run, directory in (("mixed", mixed, args.mixed_run), ("fixed_T1", fixed, args.fixed_run)):
        path = directory / run[2]["final_checkpoint"]
        checkpoints[name] = dict(path=str(path.relative_to(ROOT)),
                                 sha256=hashlib.sha256((path / "agent.th").read_bytes()).hexdigest())
        ego = load_frozen_agent(path, device=DEVICE)
        report = evaluate_ego_switch(ego, teammates, boards, schedule)
        if not torch.equal(report["stationary"]["episode_returns"], run[3]["episode_returns"]):
            raise ValueError(f"{name}: stationary scores differ from the original held-out scores by map or partner")
        reports[name] = report
    if reports["mixed"]["events"] != reports["fixed_T1"]["events"]:
        raise RuntimeError("The same PP schedule must produce identical event counts for both egos")

    differences = {
        mode: compare_ego_returns(reports["fixed_T1"][mode], reports["mixed"][mode], device=DEVICE)
        for mode in ("stationary", "switching")
    }
    effects = {name: {**report["switching"],
                      "episode_returns": report["switching"]["episode_returns"] - report["stationary"]["episode_returns"]}
               for name, report in reports.items()}
    differences["switching_minus_stationary"] = compare_ego_returns(effects["fixed_T1"], effects["mixed"], device=DEVICE)
    for report in reports.values():
        for mode in ("stationary", "switching"):
            report[mode] = serialize_scores(report[mode])
    write_json(args.output_dir / "protocol.json", dict(
        scene="pp", method="vanilla QMIX", device=DEVICE, physical_gpu_index=2, gpu_uuid=uuid,
        training_seed=config["seed"], training_steps=mixed[2]["training_steps"], checkpoints=checkpoints,
        teammates=training_protocol["pool_members"], first_map_seed=first_seed, maps_per_initial_teammate=n_maps,
        pool_order=PARTNERS, board_generator=boards["board_generator"], schedule_seed=args.schedule_seed,
        interval=list(args.interval), horizon=training_protocol["episode_limit"],
        stationary_matches_saved_scores_per_episode=True,
        design="Each map starts once with each frozen teammate. The same positions, prey policy, initial "
               "teammate and redraw schedule are used for both trained ego pairs. Before the next action after "
               "each discrete-uniform waiting time, draw uniformly from the whole pool, including the current "
               "policy. Predator bodies and ego history persist. Team size is always three.",
        source="https://proceedings.mlr.press/v216/zhang23a/zhang23a-supp.pdf#page=7",
        uncertainty="Paired bootstrap over complete maps, keeping all eight starting partners together."))
    write_json(args.output_dir / "comparison.json", dict(models=reports, fixed_minus_mixed=differences))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_run", type=project_path)
    parser.add_argument("fixed_run", type=project_path)
    parser.add_argument("output_dir", type=project_path)
    parser.add_argument("--interval", type=int, nargs=2, default=(5, 8), metavar=("MIN", "MAX"))
    parser.add_argument("--schedule-seed", type=int, default=5_300_000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    dirs = (args.mixed_run, args.fixed_run, args.output_dir)
    if len(set(dirs)) != 3 or any(OUTPUTS not in p.parents for p in dirs):
        parser.error("Use distinct project output directories")
    if args.schedule_seed < 0 or not 1 <= args.interval[0] <= args.interval[1]:
        parser.error("Use a nonnegative schedule seed and positive ordered waiting time bounds")
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
