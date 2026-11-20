import unittest

import numpy as np

from fastap_scenarios.cooperative_navigation import CooperativeNavigationWorld


class CooperativeNavigationTests(unittest.TestCase):
    def test_four_landmarks_and_closest_active_agent_reward(self):
        world = CooperativeNavigationWorld(seed=3)
        world.reset(2)
        corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for agent, landmark, pos in zip(world.agents, world.landmarks, corners):
            agent.position[:] = pos
            landmark.position[:] = pos
        self.assertEqual(world.step([0] * 4)[0], 0)
        world.set_teammate_count(1)
        reward, done, info = world.step([0] * 3)
        self.assertEqual(len(world.landmarks), 4)
        self.assertAlmostEqual(reward, -2)
        self.assertFalse(done)
        self.assertEqual(info["agent_collisions"], 0)

    def test_distinct_agent_collision_penalized_once_after_movement(self):
        world = CooperativeNavigationWorld(seed=4, collision_penalty=2.5)
        world.reset(1)
        world.agents[0].position[:] = (0, 0)
        world.agents[1].position[:] = (0.29, 0)
        world.agents[2].position[:] = (1, 1)
        reward, _, info = world.step([2, 1, 0])
        reference = sum(min(np.linalg.norm(a.position - target.position) for a in world.agents)
                        for target in world.landmarks)
        self.assertEqual(info["agent_collisions"], 1)
        self.assertLess(world.agents[0].position[0], 0.05)
        self.assertAlmostEqual(reward, -reference - 2.5)

    def test_slot_mask_and_relative_observations_survive_roster_changes(self):
        world = CooperativeNavigationWorld(seed=5)
        world.reset(1)
        world.step([0, 1, 0])
        before = world.state().copy()
        observations = world.observations()
        self.assertEqual(observations.shape, (3, 21))
        np.testing.assert_allclose(observations[0, 4:6], world.landmarks[0].position - world.agents[0].position)
        np.testing.assert_allclose(observations[0, 12:14], world.agents[1].position - world.agents[0].position)
        np.testing.assert_array_equal(observations[:, -1], 0)
        world.set_teammate_count(2)
        after = world.state()
        np.testing.assert_array_equal(before[:15], after[:15])   # retained physical agent slots
        np.testing.assert_array_equal(before[20:], after[20:])   # all four landmarks
        self.assertEqual(after[19], 1)
        np.testing.assert_array_equal(world.observations()[:, -1], 1)
        world.set_teammate_count(1)
        np.testing.assert_array_equal(world.state(), before)
        self.assertEqual(world.current_step, 1)

    def test_actions_damping_timeout_and_invalid_actions_are_atomic(self):
        world = CooperativeNavigationWorld(seed=6, episode_limit=2)
        world.reset(1)
        for agent, pos in zip(world.agents, [(0, 0), (-1, 1), (-1, -1)]):
            agent.position[:] = pos
        start = world.state().copy()
        for invalid in ([1, 2], [0, 1, 5]):
            with self.assertRaises(ValueError):
                world.step(invalid)
        np.testing.assert_array_equal(start, world.state())
        self.assertFalse(world.step([2, 0, 0])[1])
        np.testing.assert_allclose(world.agents[0].position, (0.05, 0))
        _, done, info = world.step([0] * 3)
        self.assertTrue(done)
        self.assertTrue(info["episode_limit"])
        np.testing.assert_allclose(world.agents[0].position, (0.0875, 0))
        with self.assertRaises(RuntimeError):
            world.step([0] * 3)


if __name__ == "__main__":
    unittest.main()
