import unittest

import numpy as np

from fastap_scenarios.predator_prey import PredatorPreyWorld, evade_predators


def stay(observation, available, rng):
    return 0


class PredatorPreyTests(unittest.TestCase):
    def test_fixed_roster_seed_and_observations(self):
        world = PredatorPreyWorld(seed=7)
        duplicate = PredatorPreyWorld(seed=7)
        world.reset(n_teammates=1)
        duplicate.reset()
        np.testing.assert_array_equal(world.state(), duplicate.state())
        self.assertEqual(world.observations().shape, (3, 16))
        self.assertEqual(world.state().shape, (20,))
        self.assertEqual(world.avail_actions().shape, (3, 5))
        self.assertGreater(world.prey.max_speed, world.players[0].max_speed)
        self.assertGreater(world.prey.acceleration, world.players[0].acceleration)

        predator = world.players[1]
        obs = world.observations()[1]
        np.testing.assert_allclose(obs[4:6], world.obstacles[0].position - predator.position)
        np.testing.assert_allclose(obs[8:10], world.players[0].position - predator.position)
        np.testing.assert_allclose(obs[12:14], world.prey.position - predator.position)
        snapshot = world.state()
        world.set_teammate_count(1)
        np.testing.assert_array_equal(world.state(), snapshot)
        with self.assertRaises(ValueError):
            world.set_teammate_count(2)
        with self.assertRaises(ValueError):
            world.reset(n_teammates=2)

    def test_per_predator_contact_rewards_without_reset(self):
        world = PredatorPreyWorld(seed=3, prey_policy=stay)
        for count in (0, 1, 2, 3):
            with self.subTest(count=count):
                world.reset()
                world.prey.position[:] = (0, 0)
                world.obstacles[0].position[:] = (.8, .8)
                world.obstacles[1].position[:] = (-.8, -.8)
                for i, (predator, direction) in enumerate(zip(world.players, ((1, 0), (-1, 0), (0, 1)))):
                    direction = np.array(direction)
                    predator.position[:] = direction * (.14 if i < count else .7)
                    if i < count:
                        predator.velocity[:] = -.6 * direction
                reward, done, info = world.step([0, 0, 0])
                self.assertFalse(done)
                self.assertEqual(info["contacts"], count)
                self.assertEqual(reward, 10 * count)
                self.assertEqual(info["prey_reward"], -10 * count)
                self.assertEqual(world.current_step, 1)
                self.assertLess(np.linalg.norm(world.prey.position), .01)

    def test_obstacle_changes_motion_without_tag_reward(self):
        world = PredatorPreyWorld(seed=42, prey_policy=stay)
        world.reset()
        world.prey.position[:] = (0.9, 0.9)
        world.players[0].position[:] = (0.26, 0)
        world.players[1].position[:] = (-0.8, 0.7)
        world.players[2].position[:] = (-0.7, -0.7)
        world.obstacles[0].position[:] = (0, 0)
        world.obstacles[1].position[:] = (0.7, -0.7)
        reward, _, info = world.step([0, 0, 0])
        np.testing.assert_array_equal(world.obstacles[0].position, (0, 0))
        self.assertGreater(world.players[0].velocity[0], 0)
        self.assertEqual((reward, info["prey_reward"]), (0, 0))

    def test_injected_prey_policy_and_episode_horizon(self):
        inputs = []

        def move(observation, available, rng):
            inputs.append(observation.copy())
            return 1

        world = PredatorPreyWorld(seed=3, episode_limit=2, prey_policy=move)
        world.reset()
        expected = world.prey_observation().copy()
        self.assertEqual(expected.shape, (14,))
        with self.assertRaises(ValueError):
            world.step([0, 0])
        self.assertEqual(inputs, [])
        _, done, _ = world.step([0, 0, 0])
        self.assertFalse(done)
        np.testing.assert_array_equal(inputs[0], expected)
        _, done, info = world.step([0, 0, 0])
        self.assertTrue(done and info["episode_limit"])
        self.assertEqual(len(inputs), 2)
        with self.assertRaises(RuntimeError):
            world.step([0, 0, 0])

    def test_default_prey_moves_away_from_predator(self):
        # At the origin with a predator due west, there is room to escape east.
        obs = np.array([0, 0, 0, 0, 0, .9, 0, -.9, -.3, 0, -.8, .6, -.8, -.6])
        action = evade_predators(obs, np.ones(5), np.random.default_rng(0))
        self.assertEqual(action, 2)

    def test_prey_policy_factory_isolates_worlds_and_episodes(self):
        def new_policy():
            actions = iter((2, 4))
            return lambda obs, available, rng: next(actions)

        train = PredatorPreyWorld(seed=2, prey_policy_factory=new_policy)
        test = PredatorPreyWorld(seed=3, prey_policy_factory=new_policy)
        train.reset()
        test.reset()
        for world, expected in [(train, 2), (test, 2), (train, 4)]:
            _, _, info = world.step([0, 0, 0])
            self.assertEqual(world.last_prey_action, expected)
            self.assertNotIn("last_prey_action", info)
        train.reset()
        self.assertIsNone(train.last_prey_action)
        train.step([0, 0, 0])
        self.assertEqual(train.last_prey_action, 2)
        with self.assertRaises(ValueError):
            PredatorPreyWorld(prey_policy=stay, prey_policy_factory=new_policy)


if __name__ == "__main__":
    unittest.main()
