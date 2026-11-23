"""Fastap objectives (Zhang et al., UAI 2023, Eqs. 5–11) on labeled episodes."""

from copy import deepcopy
from math import isfinite
from types import SimpleNamespace

import torch
from torch.distributions import Normal
from torch.nn import functional as F

from fastap_agents.local_q import _add_epymarl_src


def _context_loss(context, clusters, filled, previous, eta, kappa):
    """Sum compactness and -log det of one cluster kernel per encoder.

    context: [episode, time, encoder, feature]. Only the old moving average
    stops its gradient; the current mean must train the encoder via the kernel.
    """
    centers = dict(previous)
    compactness = context.new_zeros(())
    for label in clusters.unique().tolist():
        values = context[filled & (clusters == label).unsqueeze(1)]
        mean = values.mean(0)
        center = eta * previous.get(label, 0.0) + (1 - eta) * mean
        centers[label] = center
        compactness = compactness + (values - center).square().sum((-2, -1)).mean()
    # [encoder, cluster, feature], with a separate kernel for each local encoder.
    matrix = torch.stack([centers[label] for label in sorted(centers)], dim=1).double()
    distance2 = (matrix[:, :, None] - matrix[:, None, :]).square().sum(-1)
    kernel = (-kappa * distance2).exp()
    # Coincident centers can make the Gram matrix singular, especially on startup.
    jitter = 1e-8 * torch.eye(kernel.shape[-1], device=kernel.device, dtype=kernel.dtype)
    logdet = torch.linalg.slogdet(kernel + jitter).logabsdet.sum()
    loss = compactness - logdet.to(compactness.dtype)
    return loss, {key: center.detach() for key, center in centers.items()}


def _cluster_expectations(values, clusters, valid):
    """Sum each cluster's mean, giving each observed behavior type equal weight."""
    return torch.stack([values[valid & (clusters == label).unsqueeze(1)].mean()
                        for label in clusters.unique().tolist()]).sum()


class FastapLearner:
    """An injected local/global context agent, a monotonic mixer and target copies.

    The caller supplies complete padded episodes and a cluster label per episode.
    The learner doesn't instantiate a game, sample teammates, or run rollouts.
    """

    def __init__(self, agent, *, alpha_gce, alpha_lce, alpha_rec, alpha_mi=.001,
                 kappa=80., eta=.01, gamma=.99, lr=.0005, target_interval=200,
                 bootstrap_at_time_limit=False, grad_norm_clip=10.):
        if not (0 <= eta < 1 and kappa > 0 and 0 <= gamma <= 1 and target_interval >= 1
                and lr > 0 and grad_norm_clip > 0):
            raise ValueError("Invalid Fastap optimizer, kernel or target settings")
        if not all(isfinite(w) and w >= 0 for w in (alpha_gce, alpha_lce, alpha_rec, alpha_mi)):
            raise ValueError("Fastap auxiliary loss weights must be finite and nonnegative")
        self.agent = agent
        self.device = next(agent.parameters()).device
        spec = agent.spec
        _add_epymarl_src()
        from modules.mixers.qmix import QMixer
        mixer_args = SimpleNamespace(n_agents=spec.n_agents, state_shape=spec.state_dim + spec.global_dim,
                                     mixing_embed_dim=32, hypernet_layers=2, hypernet_embed=64)
        self.mixer = QMixer(mixer_args).to(self.device)
        # Repack the copied GRUs once for cuDNN's contiguous weight layout.
        self.target_agent = deepcopy(agent).to(self.device).requires_grad_(False)
        self.target_mixer = deepcopy(self.mixer).requires_grad_(False)
        self.params = list(agent.parameters()) + list(self.mixer.parameters())
        self.q_params = list(agent.q_parameters()) + list(self.mixer.parameters())
        value_ids = {id(parameter) for parameter in self.q_params}
        self.encoder_params = [parameter for parameter in self.params if id(parameter) not in value_ids]
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.alpha_gce, self.alpha_lce = alpha_gce, alpha_lce
        self.alpha_mi, self.alpha_rec = alpha_mi, alpha_rec
        self.eta, self.kappa = eta, kappa
        self.gamma = gamma
        self.bootstrap_at_time_limit = bootstrap_at_time_limit
        self.grad_norm_clip, self.target_interval = grad_norm_clip, target_interval
        self.updates = self.q_updates = 0
        self.global_centers = {}
        self.local_centers = {}

    def _td(self, batch, previous_actions, local_q, global_ctx):
        valid = batch['filled'].bool()
        actions = batch['actions'].masked_fill(~valid.unsqueeze(-1), 0)
        mixer_state = torch.cat((batch['state'], global_ctx), dim=-1)
        chosen = local_q[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        qtot = self.mixer(chosen, mixer_state[:, :-1]).squeeze(-1)
        with torch.no_grad():
            # Double-Q chooses legal actions online and evaluates them with target networks.
            next_q = local_q[:, 1:].detach().masked_fill(~batch['available'][:, 1:].bool(), -torch.inf)
            next_actions = next_q.argmax(-1, keepdim=True)
            target_local = self.target_agent.local_sequence(batch['obs'], previous_actions)
            target_q = target_local.q_values[:, 1:].gather(-1, next_actions).squeeze(-1)
            target_z = self.target_agent.global_sequence(batch['state'], previous_actions)
            target_state = torch.cat((batch['state'][:, 1:], target_z[:, 1:]), dim=-1)
            bootstrap = self.target_mixer(target_q, target_state).squeeze(-1)
            end = batch['terminated'].bool()
            if not self.bootstrap_at_time_limit:
                end = end | batch['truncated'].bool()
            target = batch['reward'] + self.gamma * bootstrap.masked_fill(end, 0)
        return (qtot[valid] - target[valid]).square().mean()

    def _step(self, loss, parameters):
        # Adam keeps independent moments and counters per tensor and skips grad=None.
        # Limiting autograd's inputs freezes every other parameter, including its moments.
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(inputs=parameters)
        torch.nn.utils.clip_grad_norm_(parameters, self.grad_norm_clip, error_if_nonfinite=True)
        self.optimizer.step()

    def update(self, batch):
        """One Algorithm 2 sweep: a Q step per cluster, then encoder/auxiliary updates.

        target_interval counts completed sweeps, independently of the number of clusters.
        """
        batch = {name: values.to(self.device) for name, values in batch.items()}
        valid = batch['filled'].bool()
        clusters = batch['cluster']
        if clusters.shape != valid.shape[:1] or (clusters < 0).any():
            raise ValueError("Pass one nonnegative cluster index per episode")
        if not valid.numel() or not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Each sampled episode needs a nonempty prefix of valid steps")
        observations_present = F.pad(valid, (1, 0), value=True)
        actions = batch['actions'].masked_fill(~valid.unsqueeze(-1), 0)
        onehot = F.one_hot(actions, self.agent.spec.n_actions).to(batch['obs'].dtype)
        previous_actions = F.pad(onehot, (0, 0, 0, 0, 1, 0))

        # Lines 14–19: apply a separate optimizer step for each sampled cluster.
        for label in clusters.unique().tolist():
            indices = clusters == label
            sample = {name: tensor[indices] for name, tensor in batch.items()}
            previous = previous_actions[indices]
            local_q = self.agent.local_sequence(sample['obs'], previous).q_values
            with torch.no_grad():
                z = self.agent.global_sequence(sample['state'], previous)
            value_loss = self._td(sample, previous, local_q, z)
            self._step(value_loss, self.q_params)
            self.q_updates += 1

        # Lines 21–22: use the newly updated Q/mixer to backpropagate the TD
        # term to their context inputs. Only encoder and auxiliary parameters step.
        local = self.agent.local_sequence(batch['obs'], previous_actions)
        global_ctx = self.agent.global_sequence(batch['state'], previous_actions)
        td = self._td(batch, previous_actions, local.q_values, global_ctx)
        gce, global_centers = _context_loss(global_ctx.unsqueeze(-2), clusters, observations_present,
                                             self.global_centers, self.eta, self.kappa)
        lce, local_centers = _context_loss(local.context, clusters, observations_present,
                                            self.local_centers, self.eta, self.kappa)
        # Eq. 6 updates the global encoder by TD + GCE; Eq. 11's MI acts on
        # the local encoder and the variational head. z is its training target.
        posterior, teammate_obs, action_logits = self.agent.auxiliary(
            local.context, local.history, global_ctx.detach())
        local_dist = Normal(local.mean, local.std)
        mi_samples = (posterior.log_prob(local.context) + local_dist.entropy()).sum((-1, -2))
        mi_bound = _cluster_expectations(mi_samples, clusters, observations_present)
        # Eq. 10 and Appendix C.1: MSE for the joint observation and NLL
        # of the joint action (product of the present teammates' categoricals).
        slot_valid = batch['teammate_mask'].bool() & valid.unsqueeze(-1)
        action_ids = batch['teammate_actions'].masked_fill(~slot_valid, 0)
        action_ids = action_ids.unsqueeze(2).expand_as(action_logits[:, :-1, :, :, 0])
        action_nll = -action_logits[:, :-1].log_softmax(-1).gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)
        mse = (teammate_obs[:, :-1] - batch['teammate_obs'].unsqueeze(2)).square().mean(-1)
        action_error = action_nll.masked_fill(~slot_valid.unsqueeze(2), 0).sum((-1, -2))
        obs_error = mse.masked_fill(~slot_valid.unsqueeze(2), 0).sum((-1, -2))
        rec_samples = action_error + obs_error / slot_valid.sum(-1).clamp_min(1)
        reconstruction = _cluster_expectations(rec_samples, clusters, valid)

        loss = (td + self.alpha_gce * gce + self.alpha_lce * lce
                - self.alpha_mi * mi_bound + self.alpha_rec * reconstruction)
        self._step(loss, self.encoder_params)
        self.global_centers, self.local_centers = global_centers, local_centers
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return dict(zip(('loss', 'td', 'gce', 'lce', 'mi_bound', 'rec'),
                        torch.stack((loss, td, gce, lce, mi_bound, reconstruction)).detach().cpu().tolist()))

    def state_dict(self):
        return dict(agent=self.agent.state_dict(), mixer=self.mixer.state_dict(),
                    target_agent=self.target_agent.state_dict(), target_mixer=self.target_mixer.state_dict(),
                    optimizer=self.optimizer.state_dict(), global_centers=self.global_centers,
                    local_centers=self.local_centers, updates=self.updates, q_updates=self.q_updates)

    def load_state_dict(self, saved):
        for key in ('agent', 'mixer', 'target_agent', 'target_mixer', 'optimizer'):
            getattr(self, key).load_state_dict(saved[key])
        self.global_centers = {key: tensor.to(self.device) for key, tensor in saved['global_centers'].items()}
        self.local_centers = {key: tensor.to(self.device) for key, tensor in saved['local_centers'].items()}
        self.updates = saved['updates']
        self.q_updates = saved.get('q_updates', self.updates)
