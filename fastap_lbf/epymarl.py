"""Register the scenario with the pinned EPyMARL checkout on sys.path."""

from envs import REGISTRY as environments
from runners import REGISTRY as runners
from runners.episode_runner import EpisodeRunner

from .env import FastapLBFEnv


class FastapEpisodeRunner(EpisodeRunner):
    def run(self, test_mode=False):
        # Upstream resets the environment inside run(), without passing test_mode.
        self.env.set_test_mode(test_mode)
        return super().run(test_mode=test_mode)


def make_env(key, **kwargs):
    # EPyMARL uses env_args.key in its checkpoint and log paths.
    if key != "fastap-lbf-6x6":
        raise ValueError(f"Unknown Fastap LBF scenario: {key}")
    return FastapLBFEnv(**kwargs)


def register_epymarl():
    environments["fastap_lbf"] = make_env
    runners["fastap_episode"] = FastapEpisodeRunner
