"""Reconstruct the Gymma-style joint observation for the formal three-player LBF pool."""

from fastap_scenarios.torch_foraging import TorchForaging


def joint_observations(states):
    """Map (..., 48) global states to (..., 63) concatenated local observations.

    Each valid state contains the board and the absolute position and level of
    every player, including at termination. Preserve all-zero sequence padding.
    """
    if states.ndim < 1 or states.shape[-1] != TorchForaging.state_shape:
        raise ValueError("Expected 48 LBF state features")
    flat = states.reshape(-1, TorchForaging.state_shape)
    result = states.new_zeros((len(flat), 3 * TorchForaging.obs_shape))
    valid = flat.ne(0).any(dim=1)
    if valid.any():
        real = flat[valid]
        players = real[:, 36:].reshape(-1, 4, 3)
        if (players[:, 3] != states.new_tensor([-1, -1, 0])).any() or (players[:, :3, 2] <= 0).any():
            raise ValueError("Expected three present players and a padded fourth slot")
        world = TorchForaging(len(real), device=states.device, n_agents=3)
        world.load_snapshot(real[:, :36].reshape(-1, 6, 6), players[:, :3, :2], players[:, :3, 2])
        result[valid] = world.observations().flatten(1).to(result.dtype)
    return result.reshape(*states.shape[:-1], 3 * TorchForaging.obs_shape)
