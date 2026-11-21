import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_methods.lbf_pool_ego_eval import ROSTERS, compare_ego_returns, evaluate_pool_ego, make_evaluation_boards
from fastap_scenarios.foraging import ForagingWorld


class LoadPolicy(torch.nn.Module):
    """A recurrent test policy with an explicit frozen slot-2 identity."""
    def __init__(self, only_last=False, wait=False):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.only_last = only_last
        self.wait = wait

    def init_hidden(self):
        return self.bias.new_zeros(1, 1)

    def forward(self, inputs, hidden):
        hidden = hidden.reshape(-1, 1)
        scores = inputs.new_zeros((len(inputs), 6))
        scores[:, 0] = 1
        if not self.wait:
            eligible = inputs[:, -1] if self.only_last else 1
            scores[:, 5] = 2 * (hidden[:, 0] > 0).float() * eligible
        return scores, hidden + 1


def arrange_map(game):
    cells = np.argwhere(game.field)
    game.field[game.field > 0] = 3
    game.initial_food_total = game.field.sum()
    r, c = cells[0]
    # Two egos and slot 2 reach the food. Slot 3 starts beyond loading range.
    positions = [(r-1, c), (r, c+1), (r+1, c)]
    far = next(pos for pos in np.ndindex(game.field.shape)
               if not game.field[pos] and abs(pos[0]-r) + abs(pos[1]-c) > 1)
    for player, pos in zip(game.players, positions+[far]):
        player.position = pos
        player.level = 1
    game._env._gen_valid_moves()


class StationaryPoolTests(unittest.TestCase):
    def test_three_and_four_player_schedules_match_independent_scalar_rollouts(self):
        start, count = 62100, 4
        boards = make_evaluation_boards(start, count)
        for n, label in enumerate(ROSTERS, start=1):
            snapshot = boards["snapshots"][label]
            for i in range(count):
                game = ForagingWorld(seed=start+i)
                try:
                    game.reset(n)
                    arrange_map(game)
                    snapshot["field"][i] = game.field
                    snapshot["levels"][i] = [p.level for p in game.players]
                    snapshot["positions"][i] = [p.position for p in game.players]
                finally:
                    game.close()

        common = dict(scene_key="lbf", obs_shape=21, n_actions=6, hidden_dim=1,
                      use_rnn=True, obs_last_action=True, obs_agent_id=True)
        ego = FrozenAgent(AgentSpec(control="ego", n_agents=2, **common), LoadPolicy())
        mate_spec = AgentSpec(control="team", n_agents=3, **common)
        teammates = dict(wait=FrozenAgent(mate_spec, LoadPolicy(wait=True)),
                         load=FrozenAgent(mate_spec, LoadPolicy(only_last=True)),
                         load2=FrozenAgent(mate_spec, LoadPolicy(only_last=True)))
        names = list(teammates)
        result = evaluate_pool_ego(ego.model, ego.spec, teammates, boards)
        self.assertEqual(result["n_episodes"], 2*count*len(names))
        self.assertEqual(len(result["per_roster"][ROSTERS[1]]["per_pairing"]), 6)
        for n, label in enumerate(ROSTERS, start=1):
            for i, name in enumerate(names):
                for j in range(count):
                    game = ForagingWorld(seed=start+j)
                    try:
                        game.reset(n)
                        arrange_map(game)
                        ids = [i] if n == 1 else [i, (i+1+j%(len(names)-1)) % len(names)]
                        policies = [ego.policy_factory(0)(), ego.policy_factory(1)()] + [
                            teammates[names[k]].policy_factory(2)() for k in ids]
                        total, done = 0, False
                        while not done:
                            actions = [policy(obs, avail, None) for policy, obs, avail
                                       in zip(policies, game.observations(), game.avail_actions())]
                            reward, done, _ = game.step(actions)
                            total += reward
                        self.assertAlmostEqual(result["episode_returns"][label][name][j], total)
                    finally:
                        game.close()
            self.assertEqual(result["episode_returns"][label]["wait"], [0] * count)
            self.assertTrue(all(x > 0 for x in result["episode_returns"][label]["load"]))
        self.assertAlmostEqual(result["mean_return"], np.mean([result["per_roster"][k]["mean_return"] for k in ROSTERS]))

    def test_comparison_resamples_whole_paired_maps_across_both_rosters(self):
        split = dict(first_map_seed=90, n_maps_per_teammate=2, pool_order=["A","B"], pairing_rule="test")
        trained = dict(**split, episode_returns={k: {"A": [1, 0], "B": [0, 1]} for k in ROSTERS})
        initial = dict(**split, episode_returns={k: {"B": [0, 0], "A": [0, 0]} for k in ROSTERS})
        expected = dict(mean_difference=.5, bootstrap_95_ci=[.5, .5])
        self.assertEqual(compare_ego_returns(trained, initial),
                         dict(**expected, per_roster={k: expected for k in ROSTERS}))
        with self.assertRaises(ValueError):
            compare_ego_returns(trained, dict(initial, first_map_seed=91))
