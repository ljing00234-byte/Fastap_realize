"""Fastap's context-conditioned Q agents and training-time encoders (appendix C.1).

Time t consumes the current observation/state and the preceding action. All
public sequence operations use batch-first tensors and zero initial memory.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FastapAgentSpec:
    obs_dim: int
    state_dim: int
    n_actions: int
    n_agents: int
    max_teammates: int
    local_dim: int
    global_dim: int
    hidden_dim: int = 64

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


class LocalMemory(NamedTuple):
    encoder: torch.Tensor
    q_network: torch.Tensor


class LocalSequence(NamedTuple):
    q_values: torch.Tensor
    context: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    history: torch.Tensor
    hidden: LocalMemory


def _gaussian(parameters):
    mean, raw_scale = parameters.chunk(2, dim=-1)
    return torch.distributions.Normal(mean, F.softplus(raw_scale) + 1e-5)


def _mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class _TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.parameters_head = nn.Linear(hidden_dim, 2 * context_dim)

    def forward(self, inputs, hidden=None):
        initial = None if hidden is None else hidden.unsqueeze(0).contiguous()
        history, final = self.gru(self.features(inputs), initial)
        return _gaussian(self.parameters_head(history)), history, final.squeeze(0)


class _LocalRole(nn.Module):
    """One ego policy that can run independently using only its own trajectory."""

    def __init__(self, spec):
        super().__init__()
        input_dim = spec.obs_dim + spec.n_actions
        self.encoder = _TrajectoryEncoder(input_dim, spec.hidden_dim, spec.local_dim)
        self.q_input = nn.Linear(input_dim + spec.local_dim, spec.hidden_dim)
        self.q_gru = nn.GRU(spec.hidden_dim, spec.hidden_dim, batch_first=True)
        self.q_head = nn.Linear(spec.hidden_dim, spec.n_actions)

    def forward(self, obs, prev_actions, hidden=None, *, sample=True):
        trajectory = torch.cat((obs, prev_actions), dim=-1)
        encoder_hidden = None if hidden is None else hidden.encoder
        dist, history, context_memory = self.encoder(trajectory, encoder_hidden)
        context = dist.rsample() if sample else dist.mean
        inputs = self.q_input(torch.cat((trajectory, context), dim=-1)).relu()
        initial = None if hidden is None else hidden.q_network.unsqueeze(0).contiguous()
        q_history, q_memory = self.q_gru(inputs, initial)
        return LocalSequence(self.q_head(q_history), context, dist.mean, dist.stddev, history,
                             LocalMemory(context_memory, q_memory.squeeze(0)))


class FastapPolicy(nn.Module):
    """Decentralized policies for the controlled agents (Algorithm 3)."""

    FORMAT = "fastap-decentralized-policy-v1"

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.roles = nn.ModuleList(_LocalRole(spec) for _ in range(spec.n_agents))

    def forward(self, obs, prev_actions, hidden=None, *, sample=True):
        """Evaluate [batch, time, ego, feature] inputs with optional [batch, ego, H] memories."""
        if obs.ndim != 4 or obs.shape[-2:] != (self.spec.n_agents, self.spec.obs_dim):
            raise ValueError("Expected obs with shape [batch, time, n_agents, obs_dim]")
        if prev_actions.shape != (*obs.shape[:-1], self.spec.n_actions):
            raise ValueError("Expected prev_actions with one action vector per observation")
        results = []
        for slot, role in enumerate(self.roles):
            memory = None if hidden is None else LocalMemory(
                hidden.encoder[:, slot], hidden.q_network[:, slot])
            results.append(role(obs[:, :, slot], prev_actions[:, :, slot], memory, sample=sample))
        return LocalSequence(
            torch.stack([r.q_values for r in results], dim=2),
            torch.stack([r.context for r in results], dim=2),
            torch.stack([r.mean for r in results], dim=2),
            torch.stack([r.std for r in results], dim=2),
            torch.stack([r.history for r in results], dim=2),
            LocalMemory(torch.stack([r.hidden.encoder for r in results], dim=1),
                        torch.stack([r.hidden.q_network for r in results], dim=1)),
        )

    @torch.no_grad()
    def act(self, obs, prev_actions, available, hidden=None, *, sample=True):
        """Take one masked greedy step; pass the returned memories on until episode reset."""
        if available.shape != prev_actions.shape or not available.bool().any(-1).all():
            raise ValueError("Each ego needs a mask with at least one legal action")
        output = self(obs.unsqueeze(1), prev_actions.unsqueeze(1), hidden, sample=sample)
        greedy = output.q_values[:, 0].masked_fill(~available.bool(), -torch.inf).argmax(-1)
        return greedy, output.hidden

    def save(self, directory):
        """Export just the local encoders and Q functions for decentralized execution."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"format": self.FORMAT, "spec": asdict(self.spec), "weights": self.state_dict()}
        path = directory / "fastap_policy.pt"
        with NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save(payload, stream)
                stream.flush()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, directory, device="cpu"):
        path = Path(directory) / "fastap_policy.pt"
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("format") != cls.FORMAT:
            raise ValueError("Unsupported Fastap policy checkpoint")
        with torch.device("meta"):
            policy = cls(FastapAgentSpec(**payload["spec"]))
        policy.load_state_dict(payload["weights"], strict=True, assign=True)
        return policy.eval().requires_grad_(False)


class FastapAgent(nn.Module):
    """Trainable policy with centralized trajectory encoder and auxiliary networks.

    This module is independent of the environment, loss functions, optimizer,
    replay buffer and CRP. Only the policy is used to choose ego actions.
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.policy = FastapPolicy(spec)
        self.global_encoder = _TrajectoryEncoder(
            spec.state_dim + spec.n_agents * spec.n_actions, spec.hidden_dim, spec.global_dim)
        self.variational = _mlp(spec.hidden_dim + spec.global_dim, spec.hidden_dim,
                                2 * spec.local_dim)
        self.obs_decoders = nn.ModuleList(
            _mlp(spec.local_dim, spec.hidden_dim, spec.max_teammates * spec.obs_dim)
            for _ in range(spec.n_agents))
        self.action_decoders = nn.ModuleList(
            _mlp(spec.local_dim, spec.hidden_dim, spec.max_teammates * spec.n_actions)
            for _ in range(spec.n_agents))

    def local_sequence(self, obs, prev_actions, hidden=None, *, sample=True):
        return self.policy(obs, prev_actions, hidden, sample=sample)

    def q_parameters(self):
        """Parameters of the individual Q functions for Algorithm 2's per-cluster TD update."""
        for role in self.policy.roles:
            for layer in (role.q_input, role.q_gru, role.q_head):
                yield from layer.parameters()

    def act(self, obs, prev_actions, available, hidden=None, *, sample=True):
        return self.policy.act(obs, prev_actions, available, hidden, sample=sample)

    def global_sequence(self, state, prev_actions, *, sample=True):
        """Encode [batch, time, state] and preceding controlled joint actions into z_t."""
        if state.ndim != 3 or state.shape[-1] != self.spec.state_dim:
            raise ValueError("Expected state with shape [batch, time, state_dim]")
        if prev_actions.shape != (*state.shape[:2], self.spec.n_agents, self.spec.n_actions):
            raise ValueError("Expected previous controlled joint actions [batch, time, ego, action]")
        inputs = torch.cat((state, prev_actions.flatten(-2)), dim=-1)
        dist, _, _ = self.global_encoder(inputs)
        return dist.rsample() if sample else dist.mean

    def auxiliary(self, local_context, local_history, global_context):
        """Return q(e|z,tau_i), teammate observation predictions and action logits.

        local_history is the local context encoder's recurrent output before
        sampling e. The last three reconstruction axes are ego, teammate, feature.
        """
        spec = self.spec
        z = global_context.unsqueeze(-2).expand(*local_history.shape[:-1], spec.global_dim)
        conditional = _gaussian(self.variational(torch.cat((z, local_history), dim=-1)))
        local_by_role = local_context.unbind(dim=-2)
        obs = torch.stack([head(local_by_role[i]) for i, head in enumerate(self.obs_decoders)], dim=-2)
        actions = torch.stack([head(local_by_role[i]) for i, head in enumerate(self.action_decoders)], dim=-2)
        return (conditional, obs.unflatten(-1, (spec.max_teammates, spec.obs_dim)),
                actions.unflatten(-1, (spec.max_teammates, spec.n_actions)))
