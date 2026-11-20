"""Register the scenario with the pinned EPyMARL checkout on sys.path."""

from envs import REGISTRY as environments
from runners import REGISTRY as runners
from runners.episode_runner import EpisodeRunner

from .registry import make_env


class FastapEpisodeRunner(EpisodeRunner):
    def run(self, test_mode=False):
        # Upstream resets the environment inside run(), without passing test_mode.
        self.env.set_test_mode(test_mode)
        return super().run(test_mode=test_mode)


def make_legacy_lbf(key, **kwargs):
    if key != "fastap-lbf-6x6":
        raise ValueError(f"Unknown Fastap LBF scenario: {key}")
    kwargs.setdefault("switch_interval", (5, 8))
    return make_env("lbf", **kwargs)


def register_epymarl():
    environments["fastap_lbf"] = make_legacy_lbf
    environments["fastap_scenario"] = make_env
    runners["fastap_episode"] = FastapEpisodeRunner
