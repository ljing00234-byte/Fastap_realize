"""A single QMIX self-play team, selected on validation and tested on the scalar LBF."""

import hashlib
import json
from math import comb
import shutil
from time import monotonic

import numpy as np

from .cooperation_validation import evaluate_cooperation
from .response_eval import evaluate_pair
from .batched_qmix import BatchedQMIX
from fastap_scenarios.torch_foraging import TorchForaging


def compare_outcomes(learned, control):
    """Exact one-sided paired tests, once per identical evaluation map."""
    learned = np.asarray(learned)
    control = np.asarray(control)
    if learned.ndim != 1 or len(learned) == 0 or learned.shape != control.shape:
        raise ValueError("Expected two equal, nonempty vectors of paired returns")
    result = {}
    for name, cutoff in (("any_food", 1/6), ("all_food", 5/6)):
        a, b = learned > cutoff, control > cutoff
        wins, losses = int((a & ~b).sum()), int((b & ~a).sum())
        n = wins + losses
        p = sum(comb(n, k) for k in range(wins, n+1)) / (2 ** n)
        result[name] = dict(trained_only=wins, control_only=losses,
                            difference=float(a.mean() - b.mean()), p_one_sided=p)
    result["mean_return_difference"] = float((learned - control).mean())
    return result


def independent_test(directory, selected, seeds):
    """Load the exported policy through the same public API used by other trainers."""
    initial = directory / "checkpoints" / "0"
    candidate = directory / "T1"
    candidate.mkdir()
    for name in ("agent.json", "agent.th"):
        shutil.copyfile(selected / name, candidate / name)
    metadata = dict(scene="lbf", slot=2, control="team", source=str(selected.relative_to(directory)),
                    sha256=hashlib.sha256((candidate / "agent.th").read_bytes()).hexdigest())
    (candidate / "role.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases = {
        "trained": (selected, candidate, False),
        "random_T1": (selected, candidate, True),
        "initial_T1": (selected, initial, False),
        "initial_team": (initial, initial, False),
    }
    game_returns = {}
    scores = {}
    for case, (egos, teammate, random) in cases.items():
        data = evaluate_pair(egos, teammate, seeds, random_teammate=random)
        values = np.array(data["episode_returns"])
        game_returns[case] = data["episode_returns"]
        scores[case] = dict(mean_return=data["mean_return"], any_food=int((values > 1/6).sum()),
                            all_food=int((values > 5/6).sum()))
    comparisons = {other: compare_outcomes(game_returns["trained"], game_returns[other])
                   for other in ("random_T1", "initial_T1", "initial_team")}
    def beats_both(metric):
        return all(comparisons[other][metric]["p_one_sided"] < 0.05 and
                   comparisons[other]["mean_return_difference"] > 0
                   for other in ("random_T1", "initial_T1"))
    return dict(checkpoint=metadata["source"], role=metadata["slot"], maps=len(seeds), scores=scores,
                paired=comparisons, evidence_for_any_food=beats_both("any_food"),
                evidence_for_full_completion=beats_both("all_food"), map_seeds=list(seeds),
                episode_returns=game_returns)


def run_single_team(args, agent_cfg, method_cfg, device, parity_check, write_json):
    directory = args.run_dir
    n_validate = args.eval_maps
    test_seeds = range(args.heldout_seed, args.heldout_seed + 1024)
    protocol = dict(start_distribution="random", fixed_teammate_slot=2, episode_limit=25,
                    training_steps=args.total_steps, training_seed=args.seed, num_envs=args.num_envs,
                    validation_seed=650000, validation_maps=n_validate, test_seed_start=test_seeds.start,
                    test_maps=len(test_seeds), validation_interval=50000,
                    selection=["full_completion_rate", "mean_return"],
                    primary_endpoint="all three foods collected within 25 steps",
                    secondary_endpoint="at least one food collected",
                    controls=["fixed egos + random T1", "fixed egos + initial T1", "initial team"],
                    evidence="positive mean return difference and one-sided paired exact p < .05 against both T1 controls")
    if set(test_seeds).intersection([args.seed, args.seed+1, args.seed+2, args.seed+10,
                                    protocol["validation_seed"]+4, protocol["validation_seed"]+5]):
        raise ValueError("Select independent held-out seeds")
    write_json(directory / "protocol.json", protocol)
    state = dict(phase="checking_environment", training_steps=0, selected_checkpoint=None)
    def update(**changes):
        state.update(changes)
        write_json(directory / "status.json", state)

    update()
    try:
        parity = parity_check(device)
        learner = BatchedQMIX(agent_cfg, method_cfg, device=device, seed=args.seed)
        world = TorchForaging(args.num_envs, device=device, seed=args.seed+10)
        checkpoints = directory / "checkpoints"
        learner.save_checkpoint(checkpoints / "0")
        update(phase="training", parity=parity)
        best = None
        next_eval = protocol["validation_interval"]
        started = monotonic()
        with (directory / "metrics.jsonl").open("x", buffering=1) as log:
            while learner.steps < args.total_steps:
                batch = learner.train_batch(world, "random")
                log.write(json.dumps(dict(kind="train", **batch), allow_nan=False) + "\n")
                if learner.steps < next_eval and learner.steps < args.total_steps:
                    continue
                result = evaluate_cooperation(learner.actor, learner.spec, "random", split="validation",
                                              split_seed=protocol["validation_seed"], device=device, n_maps=n_validate)
                log.write(json.dumps(dict(kind="validation", training_steps=learner.steps, **result)) + "\n")
                score = tuple(result["trained"][key] for key in protocol["selection"])
                if best is None or score > best:
                    best = score
                    selected = checkpoints / str(learner.steps)
                    learner.save_checkpoint(selected)
                    state["selected_checkpoint"] = str(selected.relative_to(directory))
                if learner.steps >= args.total_steps and selected.name != str(learner.steps):
                    learner.save_checkpoint(checkpoints / str(learner.steps))
                update(training_steps=learner.steps, training_episodes=learner.episodes,
                       optimizer_updates=learner.updates, validation=result["trained"],
                       elapsed_training_seconds=monotonic()-started)
                next_eval += protocol["validation_interval"]
        update(phase="independent_test")
        results = independent_test(directory, selected, test_seeds)
        write_json(directory / "heldout.json", results)
        update(phase="complete", evidence_for_full_completion=results["evidence_for_full_completion"],
               evidence_for_any_food=results["evidence_for_any_food"])
    except BaseException as error:
        update(phase="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
        raise
