"""Batched decentralized action selection and recurrent sequence evaluation."""

import torch
from torch import nn
from torch.nn import functional as F


def agent_inputs(spec, obs, last_actions):
    """Build BasicMAC inputs for any leading batch/time dimensions."""
    inputs = [obs]
    if spec.obs_last_action:
        inputs.append(last_actions)
    if spec.obs_agent_id:
        roles = torch.eye(spec.n_agents, dtype=obs.dtype, device=obs.device)
        inputs.append(roles.expand(*obs.shape[:-2], -1, -1))
    return torch.cat(inputs, dim=-1)


@torch.no_grad()
def select_actions(actor, spec, obs, available, hidden, last_actions, *, epsilon=0.0, generator=None):
    """Return legal actions [world, role] and the next independent GRU states."""
    batch_size = obs.size(0)
    inputs = agent_inputs(spec, obs, last_actions).flatten(0, 1)
    if hidden is None:
        hidden = actor.init_hidden().expand(batch_size * spec.n_agents, -1)
    q_values, hidden = actor(inputs, hidden)
    greedy = q_values.view_as(available).masked_fill(~available.bool(), -torch.inf).argmax(-1)
    # A device-side training schedule must not synchronize the host in each step.
    if isinstance(epsilon, torch.Tensor) or epsilon:
        # I.i.d. continuous scores give a uniform draw over legal actions.
        scores = torch.rand(available.shape, device=obs.device, generator=generator)
        random_actions = scores.masked_fill(~available.bool(), -1).argmax(-1)
        explore = torch.rand(greedy.shape, device=obs.device, generator=generator) < epsilon
        greedy = torch.where(explore, random_actions, greedy)
    return greedy, hidden.reshape(batch_size, spec.n_agents, -1)


class SequenceQ(nn.Module):
    """Use a single batched GRU call over time with the rollout actor's exact parameters."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        self.gru = None
        if isinstance(actor.rnn, nn.GRUCell):
            size = actor.rnn.hidden_size
            # All four parameters are supplied by the existing GRUCell. Building
            # on meta avoids allocating or randomly initializing unused weights.
            self.gru = nn.GRU(size, size, batch_first=True, device='meta')
            for name in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
                setattr(self.gru, name + '_l0', getattr(actor.rnn, name))
            self.gru.flatten_parameters()

    def forward(self, inputs):
        # Each (world, role) is one independent sequence. Keep time contiguous.
        b, t, a, _ = inputs.shape
        x = F.relu(self.actor.fc1(inputs)).transpose(1, 2).reshape(b * a, t, -1)
        if self.gru is None:
            x = F.relu(self.actor.rnn(x))
        else:
            x, _ = self.gru(x)
        return self.actor.fc2(x).reshape(b, a, t, -1).transpose(1, 2)
