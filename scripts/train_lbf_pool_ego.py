"""Train two QMIX-controlled LBF roles with one or two frozen teammates per episode."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import monotonic

from gpu2_monitor import BASELINE_PEAK, ROOT, require_monitored_cuda, run_monitored, write_json
from run_baseline import load_yaml, pool_paths, project_path


def evaluate_final_models(directory, state, members, paths, first_seed, n_maps):
    from fastap_agents import load_frozen_agent
    from fastap_methods.lbf_pool_ego_eval import compare_ego_returns, evaluate_pool_ego, make_evaluation_boards

    mates = {m["id"]: load_frozen_agent(path, device="cpu") for m, path in zip(members, paths)}
    boards = make_evaluation_boards(first_seed, n_maps)
    scores = {}
    checkpoints = {"final": state["final_checkpoint"], "initial": "checkpoints/0",
                   "validation_selected": state["selected_checkpoint"]}
    for label, path in checkpoints.items():
        if label == "validation_selected" and path == checkpoints["final"]:
            scores[label] = scores["final"]
        else:
            ego = load_frozen_agent(directory / path, device="cpu")
            scores[label] = evaluate_pool_ego(ego.model, ego.spec, mates, boards)
    paired = {name: compare_ego_returns(scores[name], scores["initial"]) for name in ("final", "validation_selected")}
    return dict(primary_checkpoint=checkpoints["final"], validation_selected_checkpoint=checkpoints["validation_selected"],
                first_seed=first_seed, shared_map_seeds_per_roster=n_maps,
                episodes_per_policy=scores["final"]["n_episodes"], scores=scores, paired=paired)


def completion_fields(heldout):
    scores = heldout["scores"]
    return dict(phase="complete", heldout_mean_return=scores["final"]["mean_return"],
                selected_heldout_mean_return=scores["validation_selected"]["mean_return"],
                initial_mean_return=scores["initial"]["mean_return"])


def recover_heldout(directory):
    with (directory / "monitor.json").open() as monitor_file:
        fcntl.flock(monitor_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = json.load(monitor_file)
        if not monitor.get("finished_at") or not isinstance(monitor.get("returncode"), int):
            raise ValueError("Wait for the training worker and its monitor to finish")
        state = json.loads((directory / "status.json").read_text())
        destination = directory / "heldout.json"
        if state["phase"] == "complete" and destination.is_file():
            print("Final evaluation already complete", flush=True)
            return
        if state["phase"] not in ("heldout_evaluation", "interrupted", "failed"):
            raise ValueError("Expected a run interrupted during final evaluation")
        config = json.loads((directory / "config.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        seed, budget = config["seed"], protocol["total_step_budget"]
        if ((config["scene"], config["control"], config["method"]["name"]) != ("lbf", "ego", "qmix")
                or budget != config["method"]["t_max"]
                or (protocol["training_seed"], protocol["training_board_seed"], protocol["training_teammate_seed"])
                != (seed, seed + 10, seed + 11)):
            raise ValueError("Inconsistent recorded training configuration or seeds")
        manifest_path = project_path(config["teammate_pool"])
        if ROOT not in manifest_path.parents:
            raise ValueError("Expected a project-local teammate pool")
        paths = pool_paths(manifest_path, "lbf")
        members = json.loads(manifest_path.read_text())["members"]
        if protocol["pool_members"] != [dict(id=m["id"], sha256=m["sha256"]) for m in members]:
            raise ValueError("Recorded pool does not match the published teammate weights")
        first_seed, n_maps = protocol["heldout_first_seed"], protocol["heldout_map_seeds_per_roster"]
        val_seed, val_maps = protocol["validation_first_seed"], protocol["validation_map_seeds_per_roster"]
        if (min(first_seed, val_seed) < 0 or min(n_maps, val_maps) < 1
                or max(first_seed, val_seed) < min(first_seed + n_maps, val_seed + val_maps)
                or 2 * len(members) * n_maps != protocol["heldout_episodes_per_policy"]):
            raise ValueError("Invalid recorded independent test maps")

        log_path = directory / "metrics.jsonl"
        rows = [json.loads(line) for line in log_path.read_text().splitlines()]
        last_train, final = rows[-2:]
        best = max((row for row in rows if row["kind"] == "validation"), key=lambda row: row["mean_return"])
        actual = final["training_steps"]
        if ((last_train["kind"], final["kind"]) != ("train", "validation") or actual != last_train["training_steps"]
                or not actual - last_train["batch_steps"] < budget <= actual
                or any(state[k] != last_train[k] for k in ("training_steps", "training_episodes", "optimizer_updates"))
                or final["first_map_seed"] != val_seed or final["n_maps_per_teammate"] != val_maps
                or final["pool_order"] != [m["id"] for m in members]
                or state["final_checkpoint"] != f"checkpoints/{actual}"
                or state["selected_checkpoint"] != f"checkpoints/{best['training_steps']}"
                or state["validation_mean_return"] != final["mean_return"]
                or state["best_validation_mean_return"] != best["mean_return"]):
            raise ValueError("Saved checkpoint selection differs from the training log")
        checkpoints = ("checkpoints/0", state["final_checkpoint"], state["selected_checkpoint"])
        hashes = {p: hashlib.sha256((directory / p / "agent.th").read_bytes()).hexdigest() for p in checkpoints}

        if destination.exists():
            # Atomic publication may have succeeded just before the worker was interrupted.
            heldout = json.loads(destination.read_text())
            if (heldout["primary_checkpoint"], heldout["validation_selected_checkpoint"], heldout["first_seed"],
                    heldout["shared_map_seeds_per_roster"], heldout["episodes_per_policy"]) != (
                    state["final_checkpoint"], state["selected_checkpoint"], first_seed, n_maps,
                    protocol["heldout_episodes_per_policy"]):
                raise ValueError("Existing held-out report conflicts with the recorded protocol")
        else:
            heldout = evaluate_final_models(directory, state, members, paths, first_seed, n_maps)
            write_json(destination, heldout)
        state.update(**completion_fields(heldout), heldout_recovery=dict(
            finished_at=datetime.now(timezone.utc).isoformat(), device="cpu", training_seed=seed,
            monitor_returncode=monitor["returncode"], monitor_error=monitor.get("error"),
            actor_sha256=hashes, metrics_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest()))
        write_json(directory / "status.json", state)
        print(f"Completed CPU held-out evaluation for seed {seed}, mean return {state['heldout_mean_return']:.6f}", flush=True)


def train(args, *, device, gpu_uuid):
    from fastap_agents import load_frozen_agent
    from fastap_methods.batched_qmix import BatchedQMIX
    from fastap_methods.lbf_pool_ego_eval import evaluate_pool_ego, make_evaluation_boards
    from fastap_scenarios.torch_lbf_ego import TorchMixedLBFEgo

    directory = args.run_dir
    paths = pool_paths(args.teammate_pool, "lbf")
    manifest = json.loads(args.teammate_pool.read_text())
    members = manifest["members"]
    names = [m["id"] for m in members]
    if len(paths) != manifest["target_size"] or len(set(names)) != len(paths) or len(paths) < 2:
        raise ValueError("Expected a complete LBF pool with at least two distinct member IDs")
    teammates = {name: load_frozen_agent(path, device=device) for name, path in zip(names, paths)}

    agent = load_yaml(ROOT / "configs/agents/drqn.yaml")
    method = load_yaml(ROOT / "configs/methods/qmix.yaml")
    validation_interval = method["save_model_interval"]
    method.update(t_max=args.total_steps, bootstrap_at_time_limit=args.bootstrap_at_time_limit, standardise_rewards=True,
                  test_interval=validation_interval, test_nepisode=2 * len(names) * args.validation_maps)
    write_json(directory / "config.json", dict(scene="lbf", control="ego", device=device,
               gpu_uuid=gpu_uuid, num_envs=args.num_envs, seed=args.seed, agent=agent, method=method,
               teammate_pool=str(args.teammate_pool.relative_to(ROOT))))
    write_json(directory / "protocol.json", dict(
        controlled_slots=[0, 1], possible_frozen_slots=[2, 3], episode_limit=TorchMixedLBFEgo.episode_limit,
        initial_weights="random", training_start_distribution="random LBF boards",
        paper="https://proceedings.mlr.press/v216/zhang23a.html",
        epymarl_reference="f355a55262ac9afecdb53368fec6337c549cc160",
        teammate_sampling="half of every rollout batch has one teammate; half has two distinct teammates; "
                          "uniform draw without replacement per episode, fixed until reset",
        local_choices=["equal proportions of the two roster sizes", "pair two independently generated single-teammate "
                       "policies without replacement", "fixed eight-member pool", "parallel environments and RNG seeds",
                       "validation and held-out map counts, pairing schedule and scoring cadence"],
        agent_architecture="QMIX supplement Fig 1(c): local observation and previous action, MLP-GRU-MLP",
        td_reward="pinned EPyMARL QMIX reward standardisation; game, replay and reported returns remain raw",
        pool_members=[dict(id=m["id"], sha256=m["sha256"]) for m in members],
        training_seed=args.seed, training_board_seed=args.seed + 10, training_teammate_seed=args.seed + 11,
        total_step_budget=args.total_steps, num_envs=args.num_envs,
        validation_first_seed=args.validation_seed, validation_map_seeds_per_roster=args.validation_maps,
        validation_episodes=method["test_nepisode"], evaluation_epsilon=method["evaluation_epsilon"],
        validation_interval=validation_interval, primary_checkpoint="final training step",
        supplementary_selection="highest stationary validation mean return, first checkpoint on a tie",
        heldout_first_seed=args.heldout_seed, heldout_map_seeds_per_roster=args.heldout_maps,
        heldout_episodes_per_policy=2 * len(names) * args.heldout_maps,
        evaluation="greedy, equally weighted roster sizes, stationary teams; identical boards across checkpoints",
        control="randomly initialized ego with the same frozen teammates and boards"))
    state = dict(phase="initializing", training_steps=0, training_episodes=0, optimizer_updates=0)

    def update(**changes):
        state.update(changes)
        write_json(directory / "status.json", state)

    update()
    try:
        world = TorchMixedLBFEgo(args.num_envs, device, teammates=list(teammates.values()),
                                 seed=args.seed + 10, teammate_seed=args.seed + 11)
        learner = BatchedQMIX(agent, method, device=device, seed=args.seed, world_type=TorchMixedLBFEgo)
        checkpoints = directory / "checkpoints"
        learner.save_checkpoint(checkpoints / "0")
        validation_boards = make_evaluation_boards(args.validation_seed, args.validation_maps)
        best_return = float("-inf")
        selected = None
        next_validation = validation_interval
        started = monotonic()
        update(phase="training")
        with (directory / "metrics.jsonl").open("x", buffering=1) as log:
            while learner.steps < args.total_steps:
                batch = learner.train_batch(world, "random")
                log.write(json.dumps(dict(kind="train", **batch), allow_nan=False) + "\n")
                if learner.steps < next_validation and learner.steps < args.total_steps:
                    continue
                checkpoint = learner.save_checkpoint(checkpoints / str(learner.steps))
                report = evaluate_pool_ego(learner.actor, learner.spec, teammates, validation_boards)
                metrics = {k: v for k, v in report.items() if k != "episode_returns"}
                log.write(json.dumps(dict(kind="validation", training_steps=learner.steps,
                                          **metrics), allow_nan=False) + "\n")
                if report["mean_return"] > best_return:
                    best_return = report["mean_return"]
                    selected = checkpoint
                update(training_steps=learner.steps, training_episodes=learner.episodes,
                       optimizer_updates=learner.updates, validation_mean_return=report["mean_return"],
                       selected_checkpoint=str(selected.relative_to(directory)), best_validation_mean_return=best_return,
                       elapsed_training_seconds=monotonic() - started)
                next_validation = (learner.steps // validation_interval + 1) * validation_interval
        counts = dict(zip(names, world.sample_counts.cpu().tolist()))
        singles, doubles = world.roster_counts.cpu().tolist()
        if (singles != doubles or singles + doubles != learner.episodes
                or sum(counts.values()) != singles + 2 * doubles):
            raise RuntimeError("Teammate appearances or team size counts disagree with the training protocol")
        update(phase="heldout_evaluation", final_checkpoint=str(checkpoint.relative_to(directory)),
               sampling_counts=counts, roster_episodes={"one_teammate": singles, "two_teammates": doubles})
        # Re-open published models and score each on the same independent boards.
        if pool_paths(args.teammate_pool, "lbf") != paths or json.loads(args.teammate_pool.read_text()) != manifest:
            raise RuntimeError("Frozen teammate pool changed during training")
        heldout = evaluate_final_models(directory, state, members, paths, args.heldout_seed, args.heldout_maps)
        write_json(directory / "heldout.json", heldout)
        update(**completion_fields(heldout))
    except BaseException as exc:
        update(phase="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=project_path)
    parser.add_argument("--teammate-pool", type=project_path, default=ROOT / "outputs/lbf_teammate_pool_01/pool.json")
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--no-bootstrap-at-time-limit", action="store_false", dest="bootstrap_at_time_limit",
                        help="Treat the 25-step timeout as terminal in the QMIX training target")
    parser.add_argument("--validation-seed", type=int, default=2_100_000)
    parser.add_argument("--validation-maps", type=int, default=64)
    parser.add_argument("--heldout-seed", type=int, default=2_200_000)
    parser.add_argument("--heldout-maps", type=int, default=1024)
    parser.add_argument("--expected-peak-mib", type=int, default=BASELINE_PEAK)
    parser.add_argument("--finish-heldout", action="store_true",
                        help="Finish an interrupted final CPU evaluation using the run's recorded settings")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if ROOT / "outputs" not in args.run_dir.parents or ROOT not in args.teammate_pool.parents:
        parser.error("Keep the run under project outputs/ and the teammate pool inside the project")
    if not args.worker:
        if os.environ.get("FASTAP_ROOT") != str(ROOT):
            parser.error("Source scripts/epymarl-env.sh before launch")
        for variable in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
            if ROOT not in Path(os.environ.get(variable, "/")).resolve().parents:
                parser.error(f"Expected project-local {variable}")
    if args.finish_heldout:
        if any(flag.startswith("--") and flag != "--finish-heldout" for flag in sys.argv[1:]):
            parser.error("--finish-heldout reads its settings from the run directory and accepts no training flags")
        sys.path.insert(0, str(ROOT))
        import torch
        torch.set_num_threads(1)
        recover_heldout(args.run_dir)
        return
    if min(args.total_steps, args.num_envs, args.validation_maps, args.heldout_maps, args.expected_peak_mib) < 1:
        parser.error("Step, map, environment and memory counts must be positive")
    if args.num_envs % 2:
        parser.error("Use an even number of parallel environments to balance the two roster sizes")
    validation = range(args.validation_seed, args.validation_seed + args.validation_maps)
    heldout = range(args.heldout_seed, args.heldout_seed + args.heldout_maps)
    if (min(args.seed, validation.start, heldout.start) < 0 or
            max(validation.start, heldout.start) < min(validation.stop, heldout.stop)):
        parser.error("Use nonnegative seeds and disjoint validation and held-out map ranges")
    if not args.worker:
        print(f"Output {args.run_dir}; disk free {shutil.disk_usage(ROOT).free // (1024 ** 3)} GiB", flush=True)
        worker = [str(Path(__file__).resolve()), "--worker", "--expected-peak-mib",
                  str(args.expected_peak_mib), *sys.argv[1:]]
        run_monitored(args.run_dir, worker, args.expected_peak_mib)
        return
    sys.path.insert(0, str(ROOT))
    uuid = require_monitored_cuda()
    import torch
    torch.set_num_threads(1)
    train(args, device="cuda:0", gpu_uuid=uuid)


if __name__ == "__main__":
    main()
