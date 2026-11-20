"""Dispatch agent construction and checkpoint loading independently of the learner."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from .local_q import AgentSpec, FrozenAgent, _add_epymarl_src


@dataclass(frozen=True)
class AgentBackend:
    """Adapter for a trainable controller and its decentralized policy.

    build_controller(scheme, groups, args) returns an EPyMARL-compatible MAC.
    build_spec(env_info, config, *, scene_key, control) returns an object whose
    save(directory) writes agent.json containing this backend's format.
    load_frozen(directory, *, device) returns a policy with spec and
    policy_factory(original_slot). The learner owns the checkpoint directory.
    """

    checkpoint_format: str
    build_controller: Callable
    build_spec: Callable
    load_frozen: Callable


_BACKENDS = {}


def register_agent_backend(name, backend):
    """Register a configuration name and its uniquely identifiable checkpoint format."""
    if not name or not backend.checkpoint_format:
        raise ValueError("Provide a nonempty agent name and checkpoint format")
    if name in _BACKENDS or any(b.checkpoint_format == backend.checkpoint_format for b in _BACKENDS.values()):
        raise ValueError(f"Agent name or checkpoint format already registered: {name}, {backend.checkpoint_format}")
    _BACKENDS[name] = backend


def get_agent_backend(name):
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown agent {name!r}. Registered agents: {', '.join(_BACKENDS)}") from None


def load_frozen_agent(checkpoint_dir, *, device="cpu"):
    """Read only registered formats. Checkpoints cannot choose a Python import path."""
    metadata = json.loads((Path(checkpoint_dir) / "agent.json").read_text(encoding="utf-8"))
    fmt = metadata.get("format")
    for backend in _BACKENDS.values():
        if fmt == backend.checkpoint_format:
            return backend.load_frozen(checkpoint_dir, device=device)
    raise ValueError(f"No registered agent can load checkpoint format {fmt!r}")


def _build_shared_controller(scheme, groups, args):
    _add_epymarl_src()
    from controllers import REGISTRY as controllers

    return controllers[args.mac](scheme, groups, args)


register_agent_backend(
    "rnn", AgentBackend(AgentSpec.FORMAT, _build_shared_controller, AgentSpec.from_epymarl, FrozenAgent.load),
)
