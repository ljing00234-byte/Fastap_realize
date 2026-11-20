"""Compare paired PP time-limit runs and replay their selected teams on fresh starts."""

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastap_agents import load_frozen_agent
from fastap_agents.batched import agent_inputs
from fastap_methods.predator_prey_eval import paired_return_difference, rollout_contacts
from fastap_scenarios.predator_prey import PredatorPreyWorld


def read_run(directory, bootstrap):
    settings = json.loads((directory / "config.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    for section in (settings["method"], protocol):
        if section.pop("bootstrap_at_time_limit", True) != bootstrap:
            raise ValueError(f"Unexpected time-limit setting in {directory}")
    if settings["scene"] != "pp":
        raise ValueError("Expected a PP run")
    heldout = json.loads((directory / "heldout.json").read_text())
    return settings, protocol, heldout


@torch.inference_mode()
def median_start_value(checkpoint, method, obs, state):
    frozen = load_frozen_agent(checkpoint)
    from modules.mixers.qmix import QMixer
    spec = frozen.spec
    previous = obs.new_zeros((*obs.shape[:2], spec.n_actions))
    local_inputs = agent_inputs(spec, obs, previous).flatten(0, 1)
    q, _ = frozen.model(local_inputs, frozen.model.init_hidden().expand(len(local_inputs), -1))
    mixer = QMixer(SimpleNamespace(**method, n_agents=spec.n_agents, state_shape=state.shape[-1]))
    mixer.load_state_dict(torch.load(checkpoint / "mixer.th", weights_only=True, map_location="cpu"))
    values = mixer(q.amax(-1).reshape(len(obs), spec.n_agents), state)
    return values.quantile(.5).item()


def compare(baseline, finite, fresh_seed):
    runs = [read_run(baseline, True), read_run(finite, False)]
    settings, protocol, held = runs[0]
    if runs[1][:2] != (settings, protocol) or runs[1][2]["map_seeds"] != held["map_seeds"]:
        raise ValueError("Expected matched configurations, protocols and evaluation starts")
    for name in ("agent.th", "mixer.th"):
        a, b = [torch.load(d / "checkpoints/0" / name, weights_only=True, map_location="cpu")
                for d in (baseline, finite)]
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError(f"Different initialization in {name}")
    seeds = range(fresh_seed, fresh_seed + protocol["test_maps"])
    if set(seeds).intersection(held["map_seeds"] + [protocol["validation_seed"], protocol["training_seed"]]):
        raise ValueError("Choose fresh confirmation seeds")
    obs, states = [], []
    for seed in seeds:
        world = PredatorPreyWorld(seed=seed)
        world.reset()
        obs.append(world.observations())
        states.append(world.state())
        world.close()
    obs, states = (torch.from_numpy(np.stack(x)) for x in (obs, states))
    result = dict(training_seed=protocol["training_seed"], fresh_map_seeds=[seeds.start, seeds.stop-1])
    fresh_returns = []
    for directory, label, (_, _, test) in zip((baseline, finite), ("bootstrap", "finite_horizon"), runs):
        selected = directory / test["checkpoint"]
        frozen = load_frozen_agent(selected)
        roles = [frozen.policy_factory(i) for i in range(frozen.spec.n_agents)]
        returns = [int(10 * rollout_contacts(seed, roles).sum()) for seed in seeds]
        fresh_returns.append(returns)
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        training = [r for r in rows if r["kind"] == "train"]
        steps = sum(r["batch_steps"] for r in training)
        if steps != protocol["training_steps"] or training[-1]["training_steps"] != steps:
            raise ValueError(f"Incomplete training in {directory}")
        last = directory / "checkpoints" / str(steps)
        result[label] = dict(steps=steps, episodes=training[-1]["training_episodes"],
                             updates=training[-1]["optimizer_updates"], selected=test["checkpoint"],
                             heldout=test["scores"]["trained"], fresh_mean_return=float(np.mean(returns)),
                             fresh_games_with_contact=int(np.count_nonzero(returns)), fresh_returns=returns,
                             final_median_start_value=median_start_value(last, settings["method"], obs, states),
                             median_last_100k_loss=median(r["loss"] for r in training
                                                         if r["training_steps"] > steps-100000))
    result["heldout_difference"] = paired_return_difference(runs[1][2]["episode_returns"]["trained"],
                                                           held["episode_returns"]["trained"])
    result["fresh_difference"] = paired_return_difference(fresh_returns[1], fresh_returns[0])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("finite", type=Path)
    parser.add_argument("--fresh-seed", type=int, default=1400000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(compare(args.baseline, args.finite, args.fresh_seed), indent=2))
