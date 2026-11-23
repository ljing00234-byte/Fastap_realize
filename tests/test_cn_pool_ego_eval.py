import unittest

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_methods.cn_pool_ego_eval import evaluate_pool_ego, make_evaluation_boards, serialize_scores
from fastap_scenarios.cooperative_navigation import CooperativeNavigationWorld


class FixedAction(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.eye(5)[action])

    def init_hidden(self):
        return self.scores.new_zeros(1, 1)

    def forward(self, inputs, hidden):
        return self.scores.expand(len(inputs), -1), hidden


class CNEgoEvaluationTests(unittest.TestCase):
    def test_shared_boards_and_scores_match_scalar_environment(self):
        n_maps = 4
        boards = make_evaluation_boards(41, n_maps, device="cpu")
        singleton = make_evaluation_boards(43, 1)
        for name in ("positions", "velocity", "landmarks"):
            self.assertTrue(torch.equal(boards[name][2], singleton[name][0]))
        fields = dict(scene_key="cn", obs_shape=21, n_actions=5, hidden_dim=1, use_rnn=True,
                      obs_last_action=True, obs_agent_id=True)
        ego_spec = AgentSpec(control="ego", n_agents=2, **fields)
        team_spec = AgentSpec(control="team", n_agents=3, **fields)
        ego = FixedAction(0)
        mates = {str(action): FrozenAgent(team_spec, FixedAction(action)) for action in (0, 1)}
        report = evaluate_pool_ego(ego, ego_spec, mates, boards, return_device_tensors=True)
        self.assertTrue(ego.training)
        self.assertEqual(report["n_episodes"], 2 * n_maps)
        self.assertEqual(tuple(report["episode_returns"].shape), (2, n_maps))
        report = serialize_scores(report)
        for name in mates:
            scores = []
            for i in range(n_maps):
                world = CooperativeNavigationWorld(seed=i)
                try:
                    world.reset(n_teammates=1)
                    for agent, position in zip(world.agents, boards["positions"][i]):
                        agent.position[:] = position.numpy()
                    for landmark, position in zip(world.landmarks, boards["landmarks"][i]):
                        landmark.position[:] = position.numpy()
                    rewards, distances, collisions = [], [], []
                    for _ in range(world.episode_limit):
                        reward, _, info = world.step([0, 0, int(name)])
                        rewards.append(reward)
                        distances.append(info["landmark_distance_sum"])
                        collisions.append(info["agent_collisions"])
                    scores.append((sum(rewards), np.mean(distances), sum(collisions)))
                finally:
                    world.close()
            scores = np.array(scores)
            np.testing.assert_allclose(report["episode_returns"][name], scores[:, 0], rtol=1e-6)
            summary = report["per_teammate"][name]
            np.testing.assert_allclose([summary["mean_return"], summary["mean_landmark_distance_sum_per_step"],
                                        summary["mean_pair_collisions_per_episode"]], scores.mean(0), rtol=1e-6)
