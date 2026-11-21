"""Train two QMIX-controlled LBF roles with one or two frozen teammates per episode."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import monotonic

from gpu2_monitor import BASELINE_PEAK, ROOT, require_monitored_cuda, run_monitored, write_json
from run_baseline import load_yaml, pool_paths, project_path


def train(args, *, device, gpu_uuid):
    from fastap_agents import load_frozen_agent
    from fastap_methods.batched_qmix import BatchedQMIX
    from fastap_methods.lbf_pool_ego_eval import compare_ego_returns, evaluate_pool_ego, make_evaluation_boards
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
    method.update(t_max=args.total_steps, bootstrap_at_time_limit=True, standardise_rewards=True,
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
        initial = learner.save_checkpoint(checkpoints / "0")
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
        test_mates = {name: load_frozen_agent(path, device="cpu") for name, path in zip(names, paths)}
        test_boards = make_evaluation_boards(args.heldout_seed, args.heldout_maps)
        scores = {}
        for label, path in (("final", checkpoint), ("initial", initial), ("validation_selected", selected)):
            if label == "validation_selected" and path == checkpoint:
                scores[label] = scores["final"]
            else:
                ego = load_frozen_agent(path, device="cpu")
                scores[label] = evaluate_pool_ego(ego.model, ego.spec, test_mates, test_boards)
        paired = {name: compare_ego_returns(scores[name], scores["initial"])
                  for name in ("final", "validation_selected")}
        write_json(directory / "heldout.json", dict(
            primary_checkpoint=state["final_checkpoint"], validation_selected_checkpoint=state["selected_checkpoint"],
            first_seed=args.heldout_seed, shared_map_seeds_per_roster=args.heldout_maps,
            episodes_per_policy=scores["final"]["n_episodes"],
            scores=scores, paired=paired))
        update(phase="complete", heldout_mean_return=scores["final"]["mean_return"],
               selected_heldout_mean_return=scores["validation_selected"]["mean_return"],
               initial_mean_return=scores["initial"]["mean_return"])
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
    parser.add_argument("--validation-seed", type=int, default=2_100_000)
    parser.add_argument("--validation-maps", type=int, default=64)
    parser.add_argument("--heldout-seed", type=int, default=2_200_000)
    parser.add_argument("--heldout-maps", type=int, default=1024)
    parser.add_argument("--expected-peak-mib", type=int, default=BASELINE_PEAK)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if ROOT / "outputs" not in args.run_dir.parents or ROOT not in args.teammate_pool.parents:
        parser.error("Keep the run under project outputs/ and the teammate pool inside the project")
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
        if os.environ.get("FASTAP_ROOT") != str(ROOT):
            parser.error("Source scripts/epymarl-env.sh before launch")
        for variable in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
            if ROOT not in Path(os.environ.get(variable, "/")).resolve().parents:
                parser.error(f"Expected project-local {variable}")
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
