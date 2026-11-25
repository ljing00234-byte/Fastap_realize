"""Replay a fixed set of LBF starting boards for each frozen group."""

from contextlib import closing
import hashlib

from fastap_scenarios.fastap_rollout import FastapRolloutEnv
from .fastap_training import collect_probe_episodes


def collect_lbf_probes_on_starts(group, rng, starts, *, use_original_egos=True):
    """Preserve the ordinary probe rollout and verify its initial state against the shared board."""
    seeds, hashes = starts['map_seeds'], starts['initial_state_sha256']
    if len(group.policies) != starts.get('n_teammates', 1):
        raise ValueError("The LBF probe group and starting boards must have the same roster size")
    if not seeds or len(seeds) != len(hashes):
        raise ValueError("Provide a state fingerprint for every shared LBF start")
    episodes = []
    for seed, expected in zip(seeds, hashes):
        with closing(FastapRolloutEnv('lbf', [group], seed=seed)) as env:
            episode, = collect_probe_episodes(env, 0, rng, 1, use_original_egos=use_original_egos)
        actual = hashlib.sha256(episode['state'][0].numpy().tobytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Shared LBF start differs for group {group.name}, seed {seed}")
        episodes.append(episode)
    return episodes
