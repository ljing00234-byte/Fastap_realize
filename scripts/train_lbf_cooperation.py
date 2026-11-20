"""Launch and track one three-player QMIX team on progressively harder LBF starts."""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic

from gpu2_monitor import BASELINE_PEAK, ROOT, require_monitored_cuda, run_monitored, write_json


def check_parity(device):
    """Compare a full CUDA rollout with the scalar reference before training."""
    import numpy as np
    import torch
    from fastap_scenarios.foraging import ForagingWorld
    from fastap_scenarios.torch_foraging import TorchForaging

    games = [ForagingWorld(seed=300 + i) for i in range(8)]
    try:
        for game in games:
            game.reset(n_teammates=1)
        # Include a guaranteed collection and a failed two-player load.
        for game in games[:2]:
            row, col = np.argwhere(game.field > 0)[0]
            for player, cell in zip(game.players, [(row-1, col), (row+1, col), (row, col-1)]):
                player.position = cell
            game._env._gen_valid_moves()
        batch = TorchForaging(len(games), device, seed=0)
        batch.load_snapshot(np.stack([g.field for g in games]),
                            [[p.position for p in g.players] for g in games],
                            [[p.level for p in g.players] for g in games])
        rng = np.random.default_rng(700)
        def compare_views():
            for actual, expected in ((batch.state(), [g.state() for g in games]),
                                     (batch.observations(), [g.observations() for g in games]),
                                     (batch.avail_actions(), [g.avail_actions() for g in games])):
                np.testing.assert_array_equal(actual.cpu().numpy(), np.stack(expected))

        compare_views()
        for t in range(batch.episode_limit):
            actions = rng.integers(0, 6, size=(len(games), 3))
            if t == 0:
                actions[0], actions[1] = [5, 5, 5], [5, 5, 0]
            expected = [g.step(a) for g, a in zip(games, actions)]
            reward, terminal, truncated = batch.step(torch.as_tensor(actions, device=device))
            np.testing.assert_allclose(reward.cpu().numpy(), [r for r, _, _ in expected], atol=1e-7)
            np.testing.assert_array_equal(terminal.cpu().numpy(),
                                          [done and not info['episode_limit'] for _, done, info in expected])
            np.testing.assert_array_equal(truncated.cpu().numpy(), [i['episode_limit'] for _, _, i in expected])
            if t == 0:
                np.testing.assert_allclose([expected[0][0], expected[1][0]], [1/3, 0])
            compare_views()
        return {"boards": len(games), "steps_per_board": batch.episode_limit, "passed": True}
    finally:
        for game in games:
            game.close()


def run_course(args, agent_cfg, method_cfg, device):
    import torch
    from fastap_methods.batched_qmix import BatchedQMIX
    from fastap_methods.cooperation_validation import STAGES, evaluate_cooperation
    from fastap_scenarios.torch_foraging import TorchForaging

    directory = args.run_dir
    state = dict(phase="checking_environment", num_envs=args.num_envs, training_steps=0,
                 training_episodes=0, optimizer_updates=0, passed_validation_stages=[])
    def update(**changes):
        state.update(changes)
        write_json(directory / "status.json", state)

    update()
    try:
        update(parity=check_parity(device))
        if getattr(args, "benchmark_only", False):
            from fastap_methods.cooperation_benchmark import benchmark
            update(phase="benchmark")
            result = benchmark(lambda: BatchedQMIX(agent_cfg, method_cfg, device=device, seed=args.seed),
                               lambda count: TorchForaging(count, device, seed=args.seed + 10))
            write_json(directory / "benchmark.json", result)
            update(phase="benchmark_complete")
            return
        learner = BatchedQMIX(agent_cfg, method_cfg, device=device, seed=args.seed)
        world = TorchForaging(args.num_envs, device, seed=args.seed + 10)
        checkpoints = directory / "checkpoints"
        learner.save_checkpoint(checkpoints / "0")
        last_ckpt = 0
        with (directory / "metrics.jsonl").open("x", buffering=1) as metrics:
            for stage, limit in zip(STAGES, args.stage_steps):
                limit = min(limit, args.total_steps - learner.steps)
                if limit <= 0:
                    break
                learner.clear_replay()
                start_step = learner.steps
                next_eval = start_step + method_cfg["test_interval"]
                best_score = None
                selected = None
                start_time = monotonic()
                update(phase="training", stage=stage, stage_steps=0, stage_limit=limit)
                while learner.steps - start_step < limit:
                    batch = learner.train_batch(world, stage)
                    metrics.write(json.dumps(dict(kind="train", stage=stage, **batch), allow_nan=False) + "\n")
                    stage_steps = learner.steps - start_step
                    due = learner.steps >= next_eval or stage_steps >= limit
                    report = None
                    if due:
                        report = evaluate_cooperation(learner.actor, learner.spec, stage, device=device,
                                                      split="validation", n_maps=args.eval_maps)
                        metrics.write(json.dumps(dict(kind="evaluation", training_steps=learner.steps,
                                                      **report), allow_nan=False) + "\n")
                        state["latest_validation"] = report
                        while next_eval <= learner.steps:
                            next_eval += method_cfg["test_interval"]
                    passed = bool(report and report["gate_passed"])
                    improved = False
                    if report is not None:
                        primary = report["primary_metric"]
                        score = (passed, report["trained"][primary], report["paired_difference"][primary],
                                 report["trained"]["mean_return"])
                        if best_score is None or score > best_score:
                            best_score, selected, improved = score, learner.steps, True
                            state["stage_best"] = {"stage": stage, "step": selected,
                                                   "metric": primary, "value": score[1]}
                    if improved or stage_steps >= limit or learner.steps - last_ckpt >= method_cfg["save_model_interval"]:
                        learner.save_checkpoint(checkpoints / str(learner.steps))
                        last_ckpt = learner.steps
                    update(training_steps=learner.steps, stage_steps=stage_steps, optimizer_updates=learner.updates,
                           training_episodes=learner.episodes, checkpoint=f"checkpoints/{last_ckpt}",
                           stage_steps_per_second=round(stage_steps / (monotonic()-start_time), 1))
                    if passed:
                        state["passed_validation_stages"].append(stage)
                        break
                if stage not in state["passed_validation_stages"]:
                    break
        selected_path = checkpoints / str(selected)
        learner.actor.load_state_dict(torch.load(selected_path / "agent.th", map_location=device, weights_only=True))
        update(phase="heldout_evaluation", selected_checkpoint=str(selected_path.relative_to(directory)))
        results = {stage: evaluate_cooperation(learner.actor, learner.spec, stage, split="heldout",
                                               split_seed=args.heldout_seed, n_maps=args.eval_maps,
                                               device=device) for stage in STAGES}
        learner.actor.load_state_dict(torch.load(checkpoints / "0" / "agent.th", map_location=device, weights_only=True))
        initial = {stage: evaluate_cooperation(learner.actor, learner.spec, stage, split="heldout",
                                               split_seed=args.heldout_seed, n_maps=args.eval_maps,
                                               device=device)["trained"] for stage in STAGES}
        report = {"checkpoint": state["selected_checkpoint"], "stages": results,
                  "initial_team_same_maps": initial,
                  "all_heldout_gates_passed": all(r["gate_passed"] for r in results.values())}
        write_json(directory / "heldout.json", report)
        update(phase="complete", all_heldout_gates_passed=report["all_heldout_gates_passed"])
    except BaseException as exc:
        update(phase="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc))
        raise


def main():
    import os
    import shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--stage-steps", type=int, nargs=3, default=(50000, 50000, 100000),
                        metavar=("ADJACENT", "APPROACH", "RANDOM"))
    parser.add_argument("--eval-maps", type=int, default=128)
    parser.add_argument("--heldout-seed", type=int, default=200000)
    parser.add_argument("--total-steps", type=int, default=200000,
                        help="aggregate training transitions across all stages (rounded up to a rollout batch)")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="time the same training workload with 1, 8 and 32 parallel boards")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--expected-peak-mib", type=int, default=BASELINE_PEAK)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.run_dir = (ROOT / args.run_dir).resolve()
    if ROOT / "outputs" not in args.run_dir.parents or min(args.num_envs, args.eval_maps, args.total_steps,
                                                          *args.stage_steps) < 1:
        parser.error("Choose a fresh directory under outputs/ and positive training sizes")
    if args.heldout_seed < 0 or abs(args.heldout_seed - 100000) < 6 or any(
            args.heldout_seed <= s < args.heldout_seed + 6 for s in (args.seed, args.seed+1, args.seed+2, args.seed+10)):
        parser.error("Use a held-out seed range separate from training and validation")
    if not args.worker:
        if os.environ.get("FASTAP_ROOT") != str(ROOT):
            parser.error("Source scripts/epymarl-env.sh before launch")
        for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
            if ROOT not in Path(os.environ.get(key, "/")).resolve().parents:
                parser.error(f"Expected project-local {key}")
        print(f"Output {args.run_dir}; disk free {shutil.disk_usage(ROOT).free // (1024 ** 3)} GiB", flush=True)
        worker = [str(Path(__file__).resolve()), "--worker", "--expected-peak-mib",
                  str(args.expected_peak_mib), *sys.argv[1:]]
        run_monitored(args.run_dir, worker, args.expected_peak_mib)
        return
    sys.path.insert(0, str(ROOT))
    uuid = require_monitored_cuda()
    import torch
    import yaml
    torch.set_num_threads(1)
    def config(name):
        return yaml.safe_load((ROOT / "configs" / name).read_text())
    agent, method = config("agents/drqn.yaml"), config("methods/qmix.yaml")
    write_json(args.run_dir / "config.json",
               dict(scene="lbf", gpu_uuid=uuid, device="cuda:0", num_envs=args.num_envs,
                    seed=args.seed, stage_step_budgets=args.stage_steps, total_step_budget=args.total_steps,
                    eval_maps=args.eval_maps, heldout_seed=args.heldout_seed, benchmark_only=args.benchmark_only,
                    agent=agent, method=method))
    run_course(args, agent, method, "cuda:0")


if __name__ == "__main__":
    main()
