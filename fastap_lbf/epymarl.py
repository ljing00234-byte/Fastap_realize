"""Compatibility entry point for EPyMARL registration."""

from fastap_scenarios.epymarl import FastapEpisodeRunner, make_legacy_lbf as make_env, register_epymarl

__all__ = ["FastapEpisodeRunner", "make_env", "register_epymarl"]
