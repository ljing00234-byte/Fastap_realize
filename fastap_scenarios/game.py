"""Contract for a game independently of role assignment and learning algorithms."""

from typing import Protocol, Sequence, Union

import numpy as np


class GameWorld(Protocol):
    """Slots 0 and 1 are persistent egos, followed by the active teammates."""

    obs_shape: int
    state_shape: int
    n_actions: int
    episode_limit: int
    current_step: int

    @property
    def n_teammates(self) -> int: ...

    def reset(self, n_teammates: int) -> None: ...
    def set_teammate_count(self, n_teammates: int) -> None: ...
    def observations(self) -> np.ndarray: ...
    def avail_actions(self) -> np.ndarray: ...
    def state(self) -> np.ndarray: ...

    def step(self, joint_actions: Sequence[int]) -> tuple[float, bool, dict[str, Union[float, int, bool]]]:
        """Return one team reward, episode end, and numeric diagnostics including episode_limit."""
        ...

    def close(self) -> None: ...
