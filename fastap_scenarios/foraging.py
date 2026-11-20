"""Fastap's LBF task (§8.2), using lbforaging 1.0.15 for movement and loading."""

from itertools import combinations, product
from operator import index

import numpy as np
from lbforaging.foraging.environment import Action, ForagingEnv, Player


# Upstream food placement uses interior cells and separates foods by more than
# two Manhattan steps. Sampling complete layouts guarantees exactly three foods.
_FOOD_LAYOUTS = tuple(
    cells for cells in combinations(product(range(1, 5), repeat=2), 3)
    if all(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 2 for a, b in combinations(cells, 2))
)
_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


class _Board(ForagingEnv):
    def _make_obs(self, player):
        obs = super()._make_obs(player)
        # The upstream border test assumes a full 3x3 crop even at (0, 0).
        h, w = obs.field.shape
        return obs._replace(players=[p for p in obs.players
                                     if 0 <= p.position[0] < h and 0 <= p.position[1] < w])

    def _make_gym_obs(self, observations):
        obs, rewards, dones, info = super()._make_gym_obs(observations)
        if len(self.players) == 3:
            for local in obs:
                local[-3:] = (-1, -1, 0)
        return obs, rewards, dones, info


class ForagingWorld:
    """Two controlled players followed by one or two teammates in slot order.

    Local observations use the upstream food/player triples (self first). The
    centralized state contains the entire food grid and four absolute player
    triples. An absent player has coordinates (-1, -1) and level zero.
    """

    obs_shape = 21
    state_shape = 48
    n_actions = 6
    episode_limit = 25

    def __init__(self, seed=None):
        self._env = _Board(players=4, max_player_level=3, field_size=(6, 6),
                           max_food=3, sight=1, max_episode_steps=self.episode_limit,
                           force_coop=True)
        self._env.seed(seed)
        self._started = False

    @property
    def field(self):
        return self._env.field

    @property
    def players(self):
        return self._env.players

    @property
    def current_step(self):
        return self._env.current_step

    @property
    def n_teammates(self):
        return len(self.players) - 2

    @property
    def initial_food_total(self):
        return self._env._food_spawned

    @initial_food_total.setter
    def initial_food_total(self, value):
        self._env._food_spawned = value

    def _add_player(self, level):
        occupied = {p.position for p in self.players}
        empty = [cell for cell in np.ndindex(self.field.shape)
                 if self.field[cell] == 0 and cell not in occupied]
        cell = empty[self._env.np_random.randint(len(empty))]
        player = Player()
        player.setup(cell, int(level), self.field.shape)
        self.players.append(player)

    @staticmethod
    def _count(value):
        count = index(value)
        if isinstance(value, (bool, np.bool_)) or count not in (1, 2):
            raise ValueError("There must be one or two teammates")
        return count

    def reset(self, n_teammates=1):
        count = self._count(n_teammates)
        board = self._env
        rng = board.np_random
        self.field.fill(0)
        self.players.clear()
        # §8.2 gives a lower bound but no level distribution. Use levels 1..3
        # and set the foods at the stated minimum for the starting roster.
        self._levels = rng.randint(1, 4, size=4)
        food_level = sum(sorted(self._levels[:2 + count])[:3])
        for cell in _FOOD_LAYOUTS[rng.randint(len(_FOOD_LAYOUTS))]:
            self.field[cell] = food_level
        self.initial_food_total = int(self.field.sum())
        for level in self._levels[:2 + count]:
            self._add_player(level)
        board.n_agents = len(self.players)
        board.current_step = 0
        board._game_over = False
        board._gen_valid_moves()
        self._started = True

    def _require_active(self):
        if not self._started or self._env.game_over:
            raise RuntimeError("Reset the world before interacting with it")

    def set_teammate_count(self, n_teammates):
        """Only slot 3 joins/leaves. A joining player spawns on a free square."""
        self._require_active()
        count = self._count(n_teammates)
        if count == self.n_teammates:
            return
        if count == 2:
            self._add_player(self._levels[3])
        else:
            self.players.pop()
        self._env.n_agents = len(self.players)
        self._env._gen_valid_moves()

    def observations(self):
        local = [self._env._make_obs(p) for p in self.players]
        return np.stack(self._env._make_gym_obs(local)[0])

    def state(self):
        triples = np.tile(np.array([-1, -1, 0], dtype=np.float32), (4, 1))
        for slot, player in enumerate(self.players):
            triples[slot] = (*player.position, player.level)
        return np.concatenate((self.field.ravel(), triples.ravel())).astype(np.float32)

    def avail_actions(self):
        return np.asarray([[int(a in self._env._valid_actions[p]) for a in Action]
                           for p in self.players], dtype=np.int32)

    def _collision_safe_actions(self, actions):
        # If two moves fail at one destination, their original cells remain
        # occupied. Propagate the blockage to agents moving into those cells.
        if len(actions) != len(self.players):
            raise ValueError("One action is required for every active player")
        moves = []
        for p, value in zip(self.players, actions):
            action = Action(index(value))
            moves.append(action.value if action in self._env._valid_actions[p] else 0)
        while True:
            targets = [(p.position[0] + _DELTAS[a][0], p.position[1] + _DELTAS[a][1])
                       for p, a in zip(self.players, moves)]
            blocked = [i for i, target in enumerate(targets)
                       if target != self.players[i].position and targets.count(target) > 1]
            if not blocked:
                return moves
            for i in blocked:
                moves[i] = Action.NONE.value

    def step(self, joint_actions):
        self._require_active()
        before = int(self.field.sum())
        self._env.step(self._collision_safe_actions(joint_actions))
        after = int(self.field.sum())
        # Equation 23: shared team credit counts each collected food once.
        reward = (before - after) / self.initial_food_total
        return reward, bool(self._env.game_over), {
            "episode_limit": self.current_step >= self.episode_limit and after != 0,
        }

    def close(self):
        self._env.close()
