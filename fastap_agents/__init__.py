"""Shared local Q networks and frozen per-role policies for decentralized play."""

from .local_q import AgentSpec, FrozenAgent
from .registry import AgentBackend, get_agent_backend, load_frozen_agent, register_agent_backend

__all__ = ["AgentBackend", "AgentSpec", "FrozenAgent", "get_agent_backend", "load_frozen_agent",
           "register_agent_backend"]
