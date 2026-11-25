"""Select natural, shared LBF starting boards before observing any teammate policy."""

from contextlib import closing
import hashlib

import numpy as np

from fastap_scenarios.registry import make_env


CARDINAL_DELTAS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def pre_load_moves(obs, available):
    """Legal N/S/W/E moves toward nearest visible food at a genuine pre-LOAD choice."""
    food = np.asarray(obs)[:9].reshape(3, 3)
    cells = food[food[:, 2] > 0, :2]
    legal = np.asarray(available, dtype=bool)[1:5]
    if available[5] or not len(cells):
        return np.zeros(4, dtype=bool)
    own = np.asarray(obs)[9:11]
    current = np.abs(own - cells).sum(axis=1).min()
    destinations = own + CARDINAL_DELTAS
    closer = np.abs(destinations[:, None, :] - cells).sum(axis=2).min(axis=1) < current
    # Waiting (NONE) cannot by itself turn a forced direction into a choice.
    return legal & closer & (legal & ~closer).any()


def select_probe_starts(first_seed, count, *, mode='shared_preload', max_candidates=10000, n_teammates=1):
    """Screen first resets of fixed-roster boards in ascending map-seed order.

    The full-team adapter uses the same world stream as FastapRolloutEnv, while
    exposing all slots and requiring no actors. No simulator steps are taken.
    """
    if mode not in ('shared_random', 'shared_preload'):
        raise ValueError("Expected shared_random or shared_preload")
    if (type(first_seed) is not int or first_seed < 0 or type(count) is not int or count < 1 or
            type(max_candidates) is not int or max_candidates < count):
        raise ValueError("Use a nonnegative first seed and a positive count within the candidate budget")
    if type(n_teammates) is not int or n_teammates not in (1, 2):
        raise ValueError("LBF probe starts require one or two teammates")
    roles = list(range(2, 2 + n_teammates))
    seeds, hashes, seen = [], [], set()
    eligible_count = 0
    for offset in range(max_candidates):
        seed = first_seed + offset
        with closing(make_env('lbf', control='team', seed=seed, n_teammates=n_teammates)) as env:
            env.set_test_mode(True)
            obs, state = env.reset()
            available = env.get_avail_actions()
            eligible = all(pre_load_moves(obs[role], available[role]).any() for role in roles)
        eligible_count += int(eligible)
        if mode == 'shared_preload' and not eligible:
            continue
        fingerprint = hashlib.sha256(state.tobytes()).hexdigest()
        if fingerprint in seen:
            continue
        seeds.append(seed)
        hashes.append(fingerprint)
        seen.add(fingerprint)
        if len(seeds) == count:
            return dict(mode=mode, n_teammates=n_teammates, target_roles=roles,
                        first_seed=first_seed, max_candidates=max_candidates,
                        candidate_count=offset + 1, eligible_candidates=eligible_count,
                        map_seeds=seeds, initial_state_sha256=hashes)
    raise ValueError(f"Found only {len(seeds)} of {count} distinct {mode} starts in {max_candidates} candidates")
