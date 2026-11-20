import unittest

import numpy as np
import torch

from fastap_scenarios.foraging import ForagingWorld
from fastap_scenarios.torch_foraging import TorchForaging


class TorchForagingTests(unittest.TestCase):
    def load_reference_games(self, scalar):
        world = TorchForaging(len(scalar), "cpu", seed=18)
        world.load_snapshot(
            np.stack([w.field for w in scalar]),
            np.array([[p.position for p in w.players] for w in scalar]),
            np.array([[p.level for p in w.players] for w in scalar]),
            np.array([w.initial_food_total for w in scalar]),
            np.array([w.current_step for w in scalar]),
        )
        return world

    def compare_views(self, world, scalar):
        np.testing.assert_array_equal(world.state(), np.stack([w.state() for w in scalar]))
        np.testing.assert_array_equal(world.observations(), np.stack([w.observations() for w in scalar]))
        np.testing.assert_array_equal(world.avail_actions(), np.stack([w.avail_actions() for w in scalar]))

    def test_25_steps_against_scalar_reference(self):
        scalar = [ForagingWorld(seed=i) for i in range(18)]
        for game in scalar:
            self.addCleanup(game.close)
            game.reset(1)
        world = self.load_reference_games(scalar)
        rng = np.random.default_rng(45)
        self.compare_views(world, scalar)
        for _ in range(25):
            actions = rng.integers(6, size=(len(scalar), 3))
            results = [w.step(a) for w, a in zip(scalar, actions)]
            reward, term, trunc = world.step(torch.from_numpy(actions))
            self.compare_views(world, scalar)
            np.testing.assert_allclose(reward, [r for r, _, _ in results])
            np.testing.assert_array_equal(term, [d and not i["episode_limit"] for _, d, i in results])
            np.testing.assert_array_equal(trunc, [i["episode_limit"] for _, _, i in results])

    def test_collision_chain_and_swaps(self):
        w = ForagingWorld(seed=12)
        self.addCleanup(w.close)
        w.reset(1)
        w.field.fill(0)
        w.field[4, 4] = 5
        w.initial_food_total = 5
        for p, cell in zip(w.players, [(2, 1), (2, 3), (2, 0)]):
            p.position = cell
        w._env._gen_valid_moves()
        world = self.load_reference_games([w])
        for moves in [(4, 3, 4), (3, 0, 4)]:
            w.step(moves)
            world.step(torch.tensor([moves]))
            self.compare_views(world, [w])

    def test_adjacent_cooperation_and_last_step_termination(self):
        world = TorchForaging(32, "cpu", seed=2)
        world.reset("adjacent")
        start = world.snapshot()
        actions = torch.full((32, 3), 5)
        actions[:, 2] = 0
        reward, term, trunc = world.step(actions)
        self.assertTrue((reward == 0).all())
        self.assertFalse((term | trunc).any())
        actions[:, 2] = 5
        reward, term, trunc = world.step(actions)
        torch.testing.assert_close(reward, torch.full((32,), 1 / 3))
        self.assertFalse((term | trunc).any())

        # Recreate the same board with the shared target as the last remaining food.
        start["field"] -= world.field
        start["steps"].fill_(24)
        world.load_snapshot(**start)
        reward, term, trunc = world.step(actions)
        torch.testing.assert_close(reward, torch.full((32,), 1 / 3))
        self.assertTrue(term.all())
        self.assertFalse(trunc.any())
        state = world.state()
        reward, _, _ = world.step(torch.ones_like(actions))
        self.assertTrue((reward == 0).all())
        torch.testing.assert_close(state, world.state())

    def test_approach_only_changes_the_starting_distance(self):
        world = TorchForaging(128, "cpu", seed=4)
        world.reset("approach")
        starts = set()
        for board, positions, levels in zip(world.field, world.positions, world.levels):
            food = board.nonzero()
            self.assertEqual(food.shape, (3, 2))
            torch.testing.assert_close(board[board != 0], levels.sum().expand(3))
            self.assertEqual(positions.unique(dim=0).size(0), 3)
            distance = (food[:, None] - positions).abs().sum(-1).sort(-1).values
            targets = distance[(distance[:, :2] == 1).all(1) & (distance[:, 2] >= 2)]
            self.assertTrue((targets[:, 2] <= 3).any())
            starts.update(targets[:, 2].tolist())
        self.assertTrue({2, 3}.issubset(starts))

    def test_seed_reproducibility_and_snapshot_copy(self):
        a, b = [TorchForaging(8, "cpu", seed=34) for _ in range(2)]
        for layout in ("random", "approach", "adjacent"):
            for w in (a, b):
                w.reset(layout)
            torch.testing.assert_close(a.state(), b.state())
            self.assertGreater(a.state().unique(dim=0).shape[0], 1)
        start = a.snapshot()
        a.step(torch.full((8, 3), 5))
        a.load_snapshot(**start)
        torch.testing.assert_close(a.state(), b.state())


if __name__ == "__main__":
    unittest.main()
