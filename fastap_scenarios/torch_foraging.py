"""Device-resident, batched LBF, using the scalar LBF game as reference."""

import torch

from .foraging import _DELTAS, _FOOD_LAYOUTS


class TorchForaging:
    """One three- or four-player game per row; done rows stay frozen until reset.

    All actions and outputs are tensors on the configured device. Time-limit truncation
    is separate from collection of all foods so the learner can bootstrap correctly.
    """

    scene_key = "lbf"
    n_agents = 3
    n_actions = 6
    obs_shape = 21
    state_shape = 48
    episode_limit = 25

    def __init__(self, batch_size, device, seed=0, *, n_agents=3):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if n_agents not in (3, 4):
            raise ValueError("LBF requires two ego agents and one or two teammates")
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        # Immutable geometry and ordering tables. Build once per batch, off the rollout path.
        coordinates = torch.cartesian_prod(torch.arange(6), torch.arange(6))
        neighbors = coordinates[:, None] + torch.tensor(_DELTAS[1:5])[None]
        valid_move = ((neighbors >= 0) & (neighbors < 6)).all(-1)
        neighbors = neighbors.clamp(0, 5)
        delta = coordinates[None, :] - coordinates[:, None]
        local = coordinates[None, :] - (coordinates[:, None] - 1).clamp_min(0)
        local = torch.where(delta.abs().amax(-1, keepdim=True) <= 1, local, -1)
        self._coordinates = coordinates.to(device)
        self._neighbors = (6 * neighbors[..., 0] + neighbors[..., 1]).to(device)
        self._valid_move = valid_move.to(device)
        self._distance = delta.abs().sum(-1).to(device)
        self._local = local.to(device)
        self._layouts = torch.tensor([[6 * row + col for row, col in layout]
                                      for layout in _FOOD_LAYOUTS], device=device)
        self._food_order = self._visibility_order((0, 1, 2)).to(device)
        self._player_order = torch.stack([
            self._visibility_order((slot,) + tuple(i for i in range(n_agents) if i != slot))
            for slot in range(n_agents)
        ]).to(device)
        self._slots = torch.arange(n_agents, device=device)
        self._bits = 1 << self._slots
        self._padding = torch.tensor([-1, -1, 0], device=device)
        self._batch = torch.arange(batch_size, device=device)
        self._other = (~torch.eye(n_agents, dtype=torch.bool)).to(device)

    @staticmethod
    def _visibility_order(priority):
        # The last index addresses the padding triple (-1, -1, 0).
        count = len(priority)
        orders = []
        for mask in range(1 << count):
            visible = [slot for slot in priority if mask & (1 << slot)]
            orders.append(visible + [count] * (count - len(visible)))
        return torch.tensor(orders)

    def _random(self, *shape):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def _approach(self, food, adjacent):
        # Move one player one or two legal moves away from its own loading square.
        # Keeping the other players stationary always leaves a route back.
        origin = adjacent[:, -1]
        cell = origin
        for distance in (2, 3):
            candidates = self._neighbors[cell]
            valid = self._valid_move[cell] & (self._distance[food[:, None], candidates] == distance)
            valid &= (self._field.gather(1, candidates) == 0)
            valid &= (candidates[:, :, None] != adjacent[:, None, :-1]).all(-1)
            choice = self._random(self.batch_size, 4).masked_fill(~valid, -1).argmax(1)
            chosen = candidates[self._batch, choice]
            accept = valid.any(1)
            if distance == 3:
                accept &= self._random(self.batch_size) < 0.5
            cell = torch.where(accept, chosen, cell)
        return torch.cat((adjacent[:, :-1], cell[:, None]), dim=1)

    def reset(self, layout="random"):
        if layout not in ("adjacent", "approach", "random"):
            raise ValueError(f"Unknown LBF starting layout: {layout}")
        size = self.batch_size
        idx = torch.randint(len(self._layouts), (size,), generator=self.generator, device=self.device)
        self._food_cells = self._layouts[idx]
        self.levels = torch.randint(1, 4, (size, self.n_agents), generator=self.generator, device=self.device)
        food_level = self.levels.sort(1).values[:, :3].sum(1, keepdim=True)
        self._food_levels = food_level.expand(-1, 3)
        self._field = torch.zeros(size, 36, device=self.device, dtype=torch.long)
        self._field.scatter_(1, self._food_cells, self._food_levels)
        if layout == "random":
            # Exchangeable random ranks sample distinct unoccupied squares in slot order.
            ranks = self._random(size, 36).masked_fill(self._field != 0, 2)
            self._cells = ranks.topk(self.n_agents, largest=False).indices
        else:
            food_index = torch.randint(3, (size,), generator=self.generator, device=self.device)
            target = self._food_cells[self._batch, food_index]
            slots = self._random(size, 4).argsort(1)[:, :self.n_agents]
            cells = self._neighbors[target].gather(1, slots)
            if layout == "approach":
                cells = self._approach(target, cells)
                cells = cells.gather(1, self._random(size, self.n_agents).argsort(1))
            self._cells = cells
        self.initial_food_total = self._food_levels.sum(1)
        self.steps = torch.zeros(size, device=self.device, dtype=torch.long)
        self._update_status()

    @property
    def field(self):
        return self._field.view(self.batch_size, 6, 6)

    @property
    def positions(self):
        return self._coordinates[self._cells]

    def snapshot(self):
        return {name: getattr(self, name).clone() for name in
                ("field", "positions", "levels", "initial_food_total", "steps")}

    def load_snapshot(self, field, positions, levels, initial_food_total=None, steps=None):
        """Load exact game states for paired evaluation and scalar/vector parity checks."""
        size = self.batch_size
        def data(value, shape):
            tensor = torch.as_tensor(value, dtype=torch.long, device=self.device)
            if tensor.shape != shape:
                raise ValueError(f"Expected {shape}, got {tuple(tensor.shape)}")
            return tensor.clone()

        self._field = data(field, (size, 6, 6)).reshape(size, 36)
        positions = data(positions, (size, self.n_agents, 2))
        self._cells = 6 * positions[..., 0] + positions[..., 1]
        self.levels = data(levels, (size, self.n_agents))
        # Row-major ordering is observable. Pad absent foods with zero-level entries.
        ids = torch.arange(36, device=self.device).expand(size, -1)
        sorted_ids = ids.masked_fill(self._field == 0, 36).sort(1).values[:, :3]
        self._food_cells = sorted_ids.clamp_max(35)
        self._food_levels = self._field.gather(1, self._food_cells).masked_fill(sorted_ids == 36, 0)
        self.initial_food_total = (self._food_levels.sum(1) if initial_food_total is None
                                   else data(initial_food_total, (size,)))
        self.steps = (torch.zeros(size, dtype=torch.long, device=self.device) if steps is None
                      else data(steps, (size,)))
        self._update_status()

    def _update_status(self):
        self.terminated = (self._food_levels == 0).all(1)
        self.truncated = (self.steps >= self.episode_limit) & ~self.terminated
        self._moves = self._neighbors[self._cells]
        neighbors_food = self._field[self._batch[:, None, None], self._moves] != 0
        in_bounds = self._valid_move[self._cells]
        wait = torch.ones_like(self._cells, dtype=torch.bool).unsqueeze(-1)
        self._available = torch.cat((wait, in_bounds & ~neighbors_food,
                                     (neighbors_food & in_bounds).any(-1, keepdim=True)), dim=-1)

    def avail_actions(self):
        return self._available

    def _visible_entities(self, cells, levels, players):
        local = self._local[self._cells[:, :, None], cells[:, None, :]]
        visible = (local[..., 0] >= 0) & (levels[:, None, :] > 0)
        mask = (visible.long() * self._bits[:cells.shape[1]]).sum(-1)
        order = self._player_order[self._slots, mask] if players else self._food_order[mask]
        triples = torch.cat((local, levels[:, None, :, None].expand(-1, self.n_agents, -1, -1)), dim=-1)
        triples = torch.cat((triples, self._padding.expand(self.batch_size, self.n_agents, 1, 3)), dim=2)
        return triples.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, 3)).flatten(2)

    def observations(self):
        food = self._visible_entities(self._food_cells, self._food_levels, False)
        agents = self._visible_entities(self._cells, self.levels, True)
        padding = self._padding.expand(self.batch_size, self.n_agents, 4 - self.n_agents, 3).flatten(2)
        return torch.cat((food, agents, padding), dim=-1).float()

    def state(self):
        triples = torch.cat((self.positions, self.levels.unsqueeze(-1)), dim=-1)
        padding = self._padding.expand(self.batch_size, 4 - self.n_agents, 3).flatten(1)
        return torch.cat((self._field, triples.flatten(1), padding), dim=-1).float()

    def step(self, actions):
        """Advance all active games, returning reward, genuine termination and timeout."""
        if actions.shape != (self.batch_size, self.n_agents):
            raise ValueError(f"Expected actions shaped ({self.batch_size}, {self.n_agents})")
        active = ~(self.terminated | self.truncated)
        valid = self._available.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        actions = torch.where(active[:, None] & valid, actions, 0)
        moving = (actions >= 1) & (actions <= 4)
        destination = self._moves.gather(2, (actions - 1).clamp(0, 3).unsqueeze(-1)).squeeze(-1)
        destination = torch.where(moving, destination, self._cells)
        # A rejected move blocks its old square. Propagate blocks along the full chain.
        for _ in range(self.n_agents):
            conflicts = ((destination[:, :, None] == destination[:, None, :]) & self._other).any(-1)
            destination = torch.where(conflicts, self._cells, destination)
        self._cells = destination

        # Legal layouts separate foods by >2 Manhattan steps; each loader touches at most one.
        adjacent = self._distance[self._cells[:, :, None], self._food_cells[:, None, :]] == 1
        strength = (((actions == 5) * self.levels)[:, :, None] * adjacent).sum(1)
        collected = (strength >= self._food_levels) & (self._food_levels > 0)
        reward = (self._food_levels * collected).sum(1).float() / self.initial_food_total.clamp_min(1)
        self._food_levels = self._food_levels.masked_fill(collected, 0)
        self._field = torch.zeros_like(self._field).scatter_add_(1, self._food_cells, self._food_levels)
        self.steps = self.steps + active
        self._update_status()
        return reward, self.terminated, self.truncated
