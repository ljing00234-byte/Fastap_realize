import unittest
from unittest.mock import patch

import numpy as np

from fastap_lbf.env import FastapLBFEnv, TeammateGroup
from fastap_lbf.world import ForagingWorld


def idle(observation, available, rng):
    return 0


class ForagingWorldTests(unittest.TestCase):
    def test_initial_boards_meet_paper_constraints(self):
        for seed in range(12):
            for count in (1, 2):
                with self.subTest(seed=seed, teammates=count):
                    world = ForagingWorld(seed=seed)
                    world.reset(n_teammates=count)
                    self.assertEqual(world.field.shape, (6, 6))
                    foods = world.field[world.field > 0]
                    self.assertEqual(len(foods), 3)
                    self.assertEqual(len(world.players), 2 + count)
                    levels = sorted(p.level for p in world.players)
                    self.assertGreaterEqual(min(foods), sum(levels[:3]))
                    self.assertLessEqual(max(foods), sum(levels))
                    self.assertEqual(len({p.position for p in world.players}), 2 + count)
                    self.assertTrue(all(world.field[p.position] == 0 for p in world.players))

    def test_three_adjacent_loaders_earn_one_team_reward(self):
        world = ForagingWorld(seed=9)
        world.reset(n_teammates=2)
        for p, pos, level in zip(world.players, [(2, 1), (1, 2), (2, 3), (5, 5)],
                                 [1, 2, 3, 3]):
            p.position = pos
            p.level = level
        world.field.fill(0)
        world.field[2, 2], world.field[1, 4], world.field[4, 4] = 6, 7, 8
        world.initial_food_total = 21
        world._env._gen_valid_moves()

        reward, done, _ = world.step([5, 5, 0, 0])
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(world.field[2, 2], 6)
        reward, done, _ = world.step([5, 5, 5, 0])
        self.assertAlmostEqual(reward, 6 / 21)
        self.assertFalse(done)
        self.assertEqual(world.field[2, 2], 0)

    def test_corner_cannot_see_players_or_food_two_cells_away(self):
        world = ForagingWorld(seed=4)
        world.reset(n_teammates=1)
        for p, pos in zip(world.players, [(0, 0), (4, 4), (0, 2)]):
            p.position = pos
        world.field.fill(0)
        world.field[2, 0] = 6
        before = world.observations()[0].copy()
        # Change the hidden teammate and hidden food, keeping the observer fixed.
        world.players[2].position = (5, 5)
        world.field[2, 0], world.field[4, 1] = 0, 6
        np.testing.assert_array_equal(before, world.observations()[0])
        world.players[2].position = (1, 1)
        self.assertFalse(np.array_equal(before, world.observations()[0]))

    def test_blocked_moves_cannot_stack_two_players_on_one_cell(self):
        world = ForagingWorld(seed=0)
        world.reset(n_teammates=1)
        # Both front players aim for (2, 2). A third tries to follow the first.
        start = [(2, 1), (2, 3), (2, 0)]
        for player, cell in zip(world.players, start):
            player.position = cell
        world.field.fill(0)
        world.field[4, 4] = 6
        world._env._gen_valid_moves()
        world.step([4, 3, 4])
        self.assertEqual([p.position for p in world.players], start)


class RoleAndScheduleTests(unittest.TestCase):
    def test_fixed_evaluation_leaves_next_training_episode_unchanged(self):
        env = FastapLBFEnv(seed=20, switch_interval=None)
        control = FastapLBFEnv(seed=20, switch_interval=None)
        for candidate in (env, control):
            candidate.reset()
        env.set_test_mode(True)
        env.reset()
        done = False
        while not done:
            _, done, _ = env.step([0, 0])
        self.assertEqual(len(env.events), 1)

        env.set_test_mode(False)
        env.reset()
        control.reset()
        self.assertEqual(env.group_name, control.group_name)
        np.testing.assert_array_equal(env.get_state(), control.get_state())

    def test_runner_actions_and_teammate_actions_reach_their_own_slots(self):
        seen = []

        def teammate(observation, available, rng):
            action = int(np.flatnonzero(available)[-1])
            seen.append((observation.copy(), action))
            return action

        env = FastapLBFEnv(seed=11, teammate_groups=[TeammateGroup('one', (lambda: teammate,))])
        env.reset()
        teammate_view = env.world.observations()[2].copy()
        controlled = [int(np.flatnonzero(mask)[0]) for mask in env.get_avail_actions()]
        controlled[1] = int(np.flatnonzero(env.get_avail_actions()[1])[-1])
        with patch.object(env.world, 'step', wraps=env.world.step) as step:
            env.step(controlled)
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0][0], teammate_view)
        joint_actions, = step.call_args.args
        self.assertEqual(list(joint_actions), controlled + [seen[0][1]])

    def test_fixed_training_and_switching_evaluation_keep_time_and_shapes(self):
        groups = [TeammateGroup('one', (lambda: idle,)),
                  TeammateGroup('two', (lambda: idle, lambda: idle))]
        env = FastapLBFEnv(seed=19, switch_interval=(3, 3), teammate_groups=groups)
        for testing in (False, True):
            env.set_test_mode(testing)
            env.reset()
            start_board = env.world.field.copy()
            controlled_start = [p.position for p in env.world.players[:2]]
            spec = env.get_env_info()
            self.assertEqual(spec['n_agents'], 2)
            self.assertEqual(spec['n_actions'], 6)
            for tick in range(1, 26):
                _, done, info = env.step([0, 0])
                self.assertEqual(env.world.current_step, tick)
                np.testing.assert_array_equal(env.world.field, start_board)
                self.assertEqual([p.position for p in env.world.players[:2]], controlled_start)
                self.assertEqual(np.asarray(env.get_obs()).shape, (2, spec['obs_shape']))
                self.assertEqual(np.asarray(env.get_state()).shape, (spec['state_shape'],))
                self.assertEqual(done, tick == 25)
            self.assertTrue(info.get('episode_limit'))
            self.assertEqual([event['step'] for event in env.events],
                             list(range(0, 25, 3)) if testing else [0])
            if testing:
                self.assertEqual({event['n_teammates'] for event in env.events}, {1, 2})


if __name__ == '__main__':
    unittest.main()
