"""Generate teammate policies on demand for Fastap Algorithm 1, line 2."""

from contextlib import closing
from copy import deepcopy
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from fastap_agents import load_frozen_agent
from fastap_scenarios.policies import TeammateGroup
from fastap_scenarios.registry import SCENARIOS, make_world
from .qmix import PROJECT, train_qmix


class OnlineTeammates:
    """Train a new all-team QMIX group at each admission, then freeze its teammates.

    For a 2+2 roster the two teammates retain training slots 2 and 3. The
    shared DRQN receives its original role ID and keeps separate per-slot memory.
    Published checkpoints and environment-step counts are available in records.
    """

    def __init__(self, scene_config, run_dir, *, seed, device, steps_per_group):
        if isinstance(scene_config, str):
            if scene_config not in SCENARIOS:
                raise ValueError(f"Unknown scene {scene_config!r}")
            scene_config = yaml.safe_load((PROJECT / 'configs/scenarios' / f'{scene_config}.yaml').read_text())
        self.scene_config = deepcopy(scene_config)
        options = self.scene_config['env_args']
        self.scene = options['key']
        scenario = SCENARIOS[self.scene]
        requested_count = options.get('n_teammates')
        self.counts = (scenario.check_count(requested_count),) if requested_count is not None else scenario.teammate_counts
        if type(steps_per_group) is not int or steps_per_group < 1:
            raise ValueError("steps_per_group must be a positive integer")
        self.run_dir = Path(run_dir).resolve()
        if PROJECT not in self.run_dir.parents:
            raise ValueError("Generated teammate artifacts must stay inside the project")
        self.seed, self.device = seed, device
        self.agent = yaml.safe_load((PROJECT / 'configs/agents/drqn.yaml').read_text())
        self.method = yaml.safe_load((PROJECT / 'configs/methods/qmix.yaml').read_text())
        with closing(make_world(self.scene, seed=seed, **options.get('world_options', {}))) as world:
            horizon = world.episode_limit
        # Even a short functional run must reach the first optimizer update.
        # The episode runner completes at least ceil(steps / horizon) episodes.
        self.method.update(t_max=steps_per_group, test_interval=steps_per_group, test_nepisode=1,
                           save_model_interval=steps_per_group,
                           batch_size=min(self.method['batch_size'], ceil(steps_per_group / horizon)),
                           buffer_size=min(self.method['buffer_size'], steps_per_group), buffer_cpu_only=True)
        self.records = []

    def __call__(self, index):
        if type(index) is not int or index < 0:
            raise ValueError("Teammate group index must be a nonnegative integer")
        count = self.counts[index % len(self.counts)]
        group_name = f'qmix_{index:04d}_{count}p'
        directory = self.run_dir / 'teammates' / group_name
        if directory.exists():
            raise FileExistsError(f"Teammate group has already been published at {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        group_seed = int(np.random.SeedSequence([self.seed, index]).generate_state(1)[0])
        scene = deepcopy(self.scene_config)
        options = scene['env_args']
        options.pop('teammate_groups', None)
        options.update(control='team', n_teammates=count, switch_interval=None)
        with TemporaryDirectory(dir=directory.parent, prefix='.training-') as temp:
            # The existing learner verifies GPU 2 monitor/UUID before CUDA allocations.
            result = train_qmix(scene_config=scene, agent_config=self.agent, method_config=self.method,
                                seed=group_seed, device=self.device, run_dir=temp)
            if result['optimizer_updates'] < 1 or result['weight_change'] <= 0:
                raise RuntimeError(f"QMIX produced no learned actor update for {group_name}")
            checkpoint = result['checkpoint']
            # In-game teammate queries use the CPU environment; the GPU training
            # copy can be released as soon as the new group has been published.
            frozen = load_frozen_agent(Path(temp) / checkpoint, device='cpu')
            if (frozen.spec.scene_key, frozen.spec.control, frozen.spec.n_agents) != (self.scene, 'team', 2 + count):
                raise ValueError("Generated actor metadata does not match the training roster")
            group = TeammateGroup(group_name, tuple(frozen.policy_factory(slot) for slot in range(2, 2 + count)))
            Path(temp).rename(directory)
        self.records.append(dict(group=group_name, n_teammates=count, seed=group_seed,
                                 training_steps=result['training_steps'], optimizer_updates=result['optimizer_updates'],
                                 checkpoint=str((directory / checkpoint).relative_to(self.run_dir))))
        return group
