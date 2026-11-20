import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fastap_agents import AgentSpec, FrozenAgent
from fastap_methods.pool_crossplay import evaluate_pool
from fastap_methods.predator_prey_eval import rollout_contacts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_lbf_pool import digest, publish
from generate_teammate_pool import train_member
from run_baseline import pool_paths


class RoleActor(torch.nn.Module):
    def __init__(self, mate_loads):
        super().__init__()
        self.mate_loads = mate_loads

    def init_hidden(self):
        return torch.zeros(1, 1)

    def forward(self, inputs, hidden):
        q = inputs.new_full((len(inputs), 6), -1)
        q[:, 0] = 1
        q[:, 5] = 2 * (inputs[:, -1] == self.mate_loads).float()
        return q, hidden


class TeammatePoolTests(unittest.TestCase):
    def test_pp_crossplay_counts_match_scalar_mixed_teams(self):
        spec = AgentSpec(scene_key="pp", control="team", n_agents=3, obs_shape=16, n_actions=5,
                         hidden_dim=1, use_rnn=False, obs_last_action=False, obs_agent_id=False)
        class Chase(torch.nn.Module):
            def init_hidden(self):
                return torch.zeros(1, 1)

            def forward(self, inputs, hidden):
                x, y = inputs[:, 12], inputs[:, 13]
                return torch.stack((torch.zeros_like(x), -x, x, -y, y), dim=1), hidden

        class Stay(Chase):
            def forward(self, inputs, hidden):
                return inputs.new_zeros((len(inputs), 5)), hidden

        agents = [FrozenAgent(spec, model) for model in (Chase(), Stay())]
        members = [dict(id=name, path=name) for name in ("chase", "stay")]
        with patch("fastap_methods.pool_crossplay.load_frozen_agent", side_effect=agents):
            result = evaluate_pool(ROOT, members, first_seed=1510000, n_maps=8)
        for i, partners in enumerate(agents):
            for j, teammate in enumerate(agents):
                roles = [partners.policy_factory(0), partners.policy_factory(1), teammate.policy_factory(2)]
                counts = np.stack([rollout_contacts(seed, roles) for seed in range(1510000, 1510008)])
                score = result["crossplay"][i][j]
                self.assertEqual(score["mean_return"], 10 * counts.sum(1).mean())
                self.assertEqual(score["any_tag"], (counts.sum(1) > 0).sum())
                self.assertEqual(score["teammate_mean_contacts"], counts[:, 2].mean())
        self.assertGreater(result["crossplay"][0][0]["mean_return"], 0)
        self.assertEqual(result["action_disagreement"], [[0, 1], [1, 0]])

    def test_pp_member_launch_inherits_the_finite_horizon(self):
        cfg = dict(scene="pp", num_envs=32, eval_maps=256, total_step_budget=2000000,
                   heldout_seed=990000, method={"bootstrap_at_time_limit": False})
        with TemporaryDirectory(dir=ROOT / ".tmp") as tmp, patch("generate_teammate_pool.subprocess.run") as run:
            run.return_value.returncode = 0
            train_member(Path(tmp), "T2", 202, cfg)
            args = run.call_args.args[0]
            self.assertEqual(args[args.index("--scene")+1], "pp")
            self.assertEqual(args[args.index("--seed")+1], "202")
            self.assertIn("--no-bootstrap-at-time-limit", args)

    def test_crossplay_replaces_only_slot_two_and_uses_shared_histories(self):
        spec = SimpleNamespace(scene_key="lbf", n_agents=3, n_actions=6, control="team",
                               obs_last_action=False, obs_agent_id=True)
        agents = [SimpleNamespace(model=RoleActor(mate), spec=spec) for mate in (False, True)]
        fields = np.zeros((2, 6, 6), dtype=int)
        for r, c in [(2, 2), (1, 4), (4, 3)]:
            fields[:, r, c] = 6
        initial = dict(field=fields, positions=[[(2, 1), (1, 2), (2, 3)]] * 2, levels=[[2, 2, 2]] * 2)
        members = [dict(id=x, path=x) for x in ("T1", "T2")]
        with patch("fastap_methods.pool_crossplay.load_frozen_agent", side_effect=agents), \
             patch("fastap_methods.pool_crossplay._initial_boards", return_value=initial):
            report = evaluate_pool(ROOT, members, 870000, 2)
        self.assertEqual([[p['any_food'] for p in row] for row in report['crossplay']], [[0, 2], [0, 0]])
        self.assertEqual(report["action_disagreement"], [[0, 1], [1, 0]])
        self.assertEqual(report["reference_decisions"], 50)

    def test_checkpoint_publication_is_repeatable_and_checks_content(self):
        with TemporaryDirectory(dir=ROOT / ".tmp") as tmp:
            pool, run = Path(tmp)/"pool", Path(tmp)/"run"
            frozen, chosen = run/"T1", run/"checkpoints/2000000"
            for d in (frozen, chosen):
                d.mkdir(parents=True)
                (d/"agent.th").write_bytes(b"weights")
                (d/"agent.json").write_text("{}")
            role = dict(slot=2, sha256=digest(chosen/"agent.th"))
            (frozen/"role.json").write_text(json.dumps(role))
            cfg = dict(scene="lbf", method={}, agent={}, total_step_budget=2000000, num_envs=32,
                       eval_maps=256, heldout_seed=720000, single_teammate=True, seed=202)
            status = dict(phase="complete", training_steps=2000000, selected_checkpoint="checkpoints/2000000")
            report = dict(role=2, checkpoint=status["selected_checkpoint"], maps=1024, scores={},
                          evidence_for_any_food=True, evidence_for_full_completion=False)
            for name, data in (("config", cfg), ("status", status), ("heldout", report)):
                (run/(name+".json")).write_text(json.dumps(data))
            pool.mkdir()
            row = publish(pool, "T2", run, 202, cfg)
            self.assertEqual(row, publish(pool, "T2", run, 202, cfg))
            manifest = pool/"pool.json"
            manifest.write_text(json.dumps(dict(format="lbf-teammate-pool-v1", scene="lbf", role=2, members=[row])))
            self.assertEqual(pool_paths(manifest, "lbf"), [pool/"members/T2"])
            cfg.update(scene="pp", method={"bootstrap_at_time_limit": False})
            (run/"config.json").write_text(json.dumps(cfg))
            report = dict(role=2, checkpoint=status["selected_checkpoint"], maps=1024, scores={},
                          evidence_for_teammate=True, evidence_for_team=True)
            (run/"heldout.json").write_text(json.dumps(report))
            pp = publish(pool, "T3", run, 202, cfg)
            self.assertTrue(pp["evidence_for_teammate"])
            manifest.write_text(json.dumps(dict(format="pp-teammate-pool-v1", scene="pp", role=2, members=[pp])))
            self.assertEqual(pool_paths(manifest, "pp"), [pool/"members/T3"])
            with self.assertRaises(ValueError):
                pool_paths(manifest, "lbf")
            (pool/"members/T3/agent.th").write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                pool_paths(manifest, "pp")
