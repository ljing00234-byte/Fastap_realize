"""Evaluate how a representation from one map predicts actions on a different map."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from fastap_methods.fastap_crp import BehaviorModel, _collate
from fastap_methods.fastap_training import _batch, _episode
from fastap_scenarios.fastap_rollout import FastapRolloutEnv, load_fastap_groups


POOL = ROOT / "outputs/fastap_lbf_formal_pool_01"
SOURCE = POOL / "crp_own_original_seed7"
FIELDS = ("state", "teammate_actions", "teammate_mask", "filled")
PARTNERS = ("fixed_T1", "own_original")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(directory, source, n_maps, support_seed, query_seed):
    """One held-out map pair per row block. Reset all three policies for every episode."""
    target = directory / "heldout.pt"
    if target.exists():
        raise FileExistsError(target)
    cfg = json.loads((source / "clustering.json").read_text())
    pool = ROOT / cfg["pool_manifest"]
    if digest(pool) != cfg["pool_manifest_sha256"]:
        raise ValueError("Frozen pool differs from the clustering run")
    groups = [g for g in load_fastap_groups("lbf", pool) if len(g.policies) == 1]
    members = [g.name for g in groups]
    if members != cfg["protocol"]["order"] or len(members) != 8:
        raise ValueError("Expected all eight training members in their original order")
    probes = load(source / "crp.pt")["crp"]["probes"]
    seen = {p[0][0].numpy().tobytes() for p in probes}
    seeds = {split: list(range(first, first+n_maps))
             for split, first in (("support", support_seed), ("query", query_seed))}
    if set(seeds["support"]) & set(seeds["query"]):
        raise ValueError("Support and query seeds must be disjoint")
    trials = {partner: {} for partner in PARTNERS}
    for split, maps in seeds.items():
        episodes = {partner: [] for partner in PARTNERS}
        for seed in maps:
            starts = set()
            for i in range(len(groups)):
                for partner in PARTNERS:
                    factories = groups[0 if partner == "fixed_T1" else i].probe_egos
                    with closing(FastapRolloutEnv("lbf", groups, seed=seed)) as env:
                        episode = _episode(env, i, None, np.random.default_rng(seed),
                                           probe_egos=tuple(factory() for factory in factories))
                    starts.add(episode["state"][0].numpy().tobytes())
                    episodes[partner].append({k: episode[k] for k in (*FIELDS, "reward")})
            if len(starts) != 1 or not starts.isdisjoint(seen):
                raise ValueError(f"Unpaired or repeated train/support/query initial state for {seed}")
            seen.update(starts)
        for partner, rows in episodes.items():
            trials[partner][split] = _batch(rows)
    data = dict(members=members, map_seeds=seeds, trials=trials, pool_sha256=digest(pool),
                training_checkpoint_sha256=digest(source / "crp.pt"),
                row_order="map_index * n_members + member_index")
    with TemporaryDirectory(dir=directory) as tmp:
        stage = Path(tmp) / "heldout.pt"
        torch.save(data, stage)
        stage.rename(target)
    return data


def fit_state_only(model_args, protocol, admissions, probes, seed):
    """Match categorical decoder initialization, replay order, minibatches and optimizer."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = BehaviorModel(**model_args, decoder="categorical")
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol["lr"])
        for admission in admissions:
            stop = admission["probe_slice"][1]
            for _ in range(protocol["crp_model_steps"]):
                indices = torch.randperm(stop)[:protocol["minibatch_size"]].tolist()
                sample = _collate([probes[i] for i in indices], "cpu")
                zeros = sample[0].new_zeros((len(indices), model.width))
                loss = model.episode_action_nll(*sample, zeros).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict(model, state_only, trials, n_members):
    """Pair every queried episode with each identity on its support map, exactly once."""
    model.eval()
    state_only.eval()
    support = trials["support"]
    query = trials["query"]
    n_rows = len(query["state"])
    if len(support["state"]) != n_rows or n_members < 2 or n_rows % n_members:
        raise ValueError("Expected matched complete blocks of support and query members")
    scores = {"correct": [], "wrong": [], "state_only": []}
    # Each context belongs to a different physical opening from every query.
    # The evaluator never passes a query action to an encoder.
    for first in range(0, n_rows, n_members):
        rows = slice(first, first+n_members)
        sample = [query[k][rows] for k in FIELDS]
        vectors = model.encode(*(support[k][rows] for k in FIELDS))
        scores["correct"].append(model.episode_action_nll(*sample, vectors))
        wrong = [model.episode_action_nll(*sample, vectors.roll(offset, dims=0))
                 for offset in range(1, n_members)]
        # Average conditional NLLs (one per wrong identity), not probabilities.
        scores["wrong"].append(torch.stack(wrong).mean(0))
        scores["state_only"].append(state_only.episode_action_nll(*sample, torch.zeros_like(vectors)))
    return {k: torch.stack(rows).double().numpy() for k, rows in scores.items()}


def summarize(scores, bootstrap_rows, map_seeds, members):
    """Resample paired whole maps with all eight teammates moving together."""
    if not all(np.isfinite(v).all() for v in scores.values()):
        raise FloatingPointError("Non-finite categorical action NLL")
    gains = {}
    for control in ("wrong", "state_only"):
        per_map = (scores[control] - scores["correct"]).mean(axis=1)
        bounds = np.quantile(per_map[bootstrap_rows].mean(axis=1), [.025, .975])
        gains[control] = dict(mean_nats=float(per_map.mean()), paired_95_ci=bounds.tolist(),
                              maps_with_positive_gain=int(np.count_nonzero(per_map > 0)))
    return dict(mean_nll_nats={k: float(v.mean()) for k, v in scores.items()}, gain=gains,
                both_lower_bounds_positive=all(c["paired_95_ci"][0] > 0 for c in gains.values()),
                nll_by_map_and_member={k: v.tolist() for k, v in scores.items()},
                axes=["paired map index", "member"], query_map_seeds=map_seeds, members=members)


def evaluate(directory, source, seeds):
    target = directory / "cross_episode_prediction.json"
    if target.exists():
        raise FileExistsError(target)
    data = load(directory / "heldout.pt")
    config = json.loads((source / "clustering.json").read_text())
    if data["training_checkpoint_sha256"] != digest(source / "crp.pt"):
        raise ValueError("Held-out independence was checked against another training set")
    reference = load(source / "crp.pt")
    probes = reference["crp"]["probes"]
    names, maps = data["members"], data["map_seeds"]["query"]
    bootstrap_seed = 87543
    bootstrap_rows = np.random.default_rng(bootstrap_seed).integers(len(maps), size=(10000, len(maps)))
    results = []
    for seed in seeds:
        state_only = fit_state_only(reference["model_args"], config["protocol"], config["admissions"], probes, seed)
        for decoder in ("paper", "categorical"):
            run = directory / decoder / f"seed{seed}"
            checkpoint = load(run / "crp.pt")
            train_cfg = json.loads((run / "clustering.json").read_text())
            if (train_cfg["protocol"]["seed"] != seed or train_cfg["protocol"]["decoder"] != decoder
                    or train_cfg["pool_manifest_sha256"] != data["pool_sha256"]):
                raise ValueError("Unexpected checkpoint seed, decoder or frozen teammate pool")
            for key in ("lr", "minibatch_size", "crp_model_steps", "order"):
                if train_cfg["protocol"][key] != config["protocol"][key]:
                    raise ValueError(f"Training controls differ on {key}")
            candidate = checkpoint["crp"]["probes"]
            if (len(candidate) != len(probes) or
                    any(not torch.equal(a, b) for row, orig in zip(candidate, probes) for a, b in zip(row, orig))):
                raise ValueError("Training probes differ from the state-only and independence reference")
            model = BehaviorModel(**checkpoint["model_args"], decoder=decoder)
            model.load_state_dict(checkpoint["crp"]["model"])
            result = dict(seed=seed, decoder=decoder, checkpoint_sha256=digest(run/"crp.pt"), partner_trials={})
            for partner in PARTNERS:
                scores = predict(model, state_only, data["trials"][partner], len(names))
                result["partner_trials"][partner] = summarize(scores, bootstrap_rows, maps, names)
            results.append(result)
    passed = all(r["partner_trials"]["fixed_T1"]["both_lower_bounds_positive"]
                 for r in results if r["decoder"] == "categorical")
    report = dict(primary="fixed_T1", secondary="own_original", seeds=seeds,
                  map_seeds=data["map_seeds"], pool_sha256=data["pool_sha256"],
                  training_checkpoint_sha256=data["training_checkpoint_sha256"],
                  heldout_sha256=digest(directory/"heldout.pt"),
                  wrong_contexts="All other seven identities on the same support map, uniformly averaged NLL.",
                  state_control="Independently optimized categorical GRU with a constant zero context; same "
                                "decoder initialization, target minibatches, Adam settings, updates and training episodes.",
                  nll="Natural log loss over six actions. Average over active slots and valid steps within each "
                      "episode, then equally over episodes. Larger control-minus-correct gain favors transfer.",
                  paper_action_scores="Normalize the six Gaussian densities at one-hot action vertices only for "
                                      "this evaluation. The published continuous formula remains its training objective.",
                  rule="Both controls must have a positive lower paired 95% map-bootstrap bound in every model "
                       "seed on fixed-T1 partner trials. These pointwise intervals are descriptive across seeds.",
                  bootstrap_resamples=len(bootstrap_rows), bootstrap_seed=bootstrap_seed,
                  primary_criterion_passed=passed, results=results)
    with TemporaryDirectory(dir=directory) as tmp:
        stage = Path(tmp) / target.name
        stage.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        stage.rename(target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "evaluate"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--maps", type=int, default=32)
    parser.add_argument("--support-seed", type=int, default=6710000)
    parser.add_argument("--query-seed", type=int, default=6720000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    args = parser.parse_args()
    directory = (ROOT / args.directory).resolve()
    source = (ROOT / args.source).resolve()
    if (os.getenv("FASTAP_ROOT") != str(ROOT) or ROOT/"outputs" not in directory.parents
            or ROOT/"outputs" not in source.parents or directory == source):
        parser.error("Source scripts/epymarl-env.sh and choose separate project output directories")
    if args.maps < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least two map pairs and distinct model seeds")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    directory.mkdir(parents=True, exist_ok=True)
    if args.phase == "collect":
        collect(directory, source, args.maps, args.support_seed, args.query_seed)
        print(directory / "heldout.pt")
    else:
        report = evaluate(directory, source, args.seeds)
        print("Paired cross-map prediction passed:", report["primary_criterion_passed"])


if __name__ == "__main__":
    main()
