"""Fastap's incremental teammate clustering (Eqs. 1–4 and Algorithm 1)."""

from math import isfinite, log

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class BehaviorModel(nn.Module):
    """Trajectory Transformer with the paper Gaussian or a categorical action decoder."""

    def __init__(self, state_dim, n_actions, n_teammates, *, decoder='paper'):
        super().__init__()
        if decoder not in ('paper', 'categorical'):
            raise ValueError("decoder must be paper or categorical")
        self.decoder = decoder
        self.n_actions, self.n_teammates = n_actions, n_teammates
        self.width = 16
        action_dim = n_actions * n_teammates
        self.token = nn.Linear(state_dim + action_dim, self.width)
        # The paper specifies the number of layers and the 16-dimensional v.
        # Use four heads, a 64-unit FFN and masked mean pooling for the readout.
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(
            self.width, nhead=4, dim_feedforward=64, dropout=0., batch_first=True)
            for _ in range(8))
        self.gru = nn.GRU(state_dim + self.width, 16, batch_first=True)
        if decoder == 'paper':
            self.gaussian = nn.Linear(16, 2 * action_dim)
        else:
            self.categorical = nn.Linear(16, action_dim)

    def encode(self, states, actions, slots, filled, *, action_mask=None):
        """Embed s0, a0, ..., sT; an optional True mask hides the corresponding action."""
        active = slots & filled.unsqueeze(-1)
        visible = active
        if action_mask is not None:
            if action_mask.shape != actions.shape or action_mask.dtype != torch.bool:
                raise ValueError("action_mask must be a boolean mask with the same shape as actions")
            visible = active & ~action_mask
        onehot = F.one_hot(actions.masked_fill(~visible, 0), self.n_actions).to(states.dtype)
        onehot = onehot.masked_fill(~visible.unsqueeze(-1), 0)
        if action_mask is not None:
            # An unknown observed action gets a uniform token; absent slots remain zero.
            onehot = onehot.masked_fill((active & action_mask).unsqueeze(-1), 1. / self.n_actions)
        previous = F.pad(onehot.flatten(-2), (0, 0, 1, 0))
        valid_states = F.pad(filled, (1, 0), value=True)
        tokens = self.token(torch.cat((states, previous), dim=-1))
        positions = torch.arange(states.shape[1], device=states.device, dtype=states.dtype)
        frequencies = torch.arange(0, self.width, 2, device=states.device, dtype=states.dtype)
        angles = positions[:, None] * (10000. ** (-frequencies / self.width))
        tokens = tokens + torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=~valid_states)
        return tokens.masked_fill(~valid_states.unsqueeze(-1), 0).sum(1) / valid_states.sum(1, keepdim=True)

    def _decode_states(self, states, behavior):
        # The final state is visible to the trajectory encoder but has no action target.
        history = torch.cat((states[:, :-1], behavior.unsqueeze(1).expand(-1, states.size(1) - 1, -1)), -1)
        decoded, _ = self.gru(history)
        return decoded

    def _gaussian_parameters(self, decoded):
        mean, scale_logits = self.gaussian(decoded).chunk(2, dim=-1)
        return mean, F.softplus(scale_logits) + 1e-4

    def action_logits(self, states, behavior):
        """One logit per action and teammate slot, shape (episode, step, slot, action)."""
        decoded = self._decode_states(states, behavior)
        if self.decoder == 'categorical':
            logits = self.categorical(decoded)
        else:
            # At a one-hot vertex e_k, the Gaussian log density differs from its
            # value at zero by (mean_k - 1/2) / scale_k². Shared offsets cancel.
            mean, scale = self._gaussian_parameters(decoded)
            logits = (mean - .5) / scale.square()
        return logits.unflatten(-1, (self.n_teammates, self.n_actions))

    def action_log_probs(self, states, actions, slots, filled, behavior):
        """Categorical log p of each observed action; absent and padded entries contribute zero."""
        active = slots & filled.unsqueeze(-1)
        logits = self.action_logits(states, behavior)
        targets = actions.masked_fill(~active, 0)
        log_probs = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return log_probs.masked_fill(~active, 0.)

    def episode_action_nll(self, states, actions, slots, filled, behavior):
        """Per-episode cross-entropy, averaged over its observed teammate actions."""
        active_counts = (slots & filled.unsqueeze(-1)).sum((1, 2))
        if (active_counts == 0).any():
            raise ValueError("Each trajectory needs at least one observed teammate action")
        log_probs = self.action_log_probs(states, actions, slots, filled, behavior)
        return -log_probs.sum((1, 2)) / active_counts

    def trajectory_log_prob(self, states, actions, slots, filled, behavior):
        """Joint log evidence for CRP admission, summed over all observed actions."""
        if self.decoder == 'categorical':
            return self.action_log_probs(states, actions, slots, filled, behavior).sum((1, 2))
        mean, scale = self._gaussian_parameters(self._decode_states(states, behavior))
        active = slots & filled.unsqueeze(-1)
        target = F.one_hot(actions.masked_fill(~active, 0), self.n_actions).to(states.dtype)
        return self._score_actions(mean, scale, target, active)

    def _score_actions(self, mean, scale, target, active):
        scores = Normal(mean, scale).log_prob(target.flatten(-2)).reshape_as(target)
        return scores.masked_fill(~active.unsqueeze(-1), 0).sum((1, 2, 3))


def _collate(probes, device):
    """Pad only the sampled probe trajectories; histories may have different horizons."""
    return [pad_sequence(series, batch_first=True).to(device) for series in zip(*probes)]


def validate_masked_action_config(decoder, weight, span):
    if decoder not in ('paper', 'categorical'):
        raise ValueError("decoder must be paper or categorical")
    if not isfinite(weight) or weight < 0:
        raise ValueError("masked_action_weight must be finite and nonnegative")
    if weight and decoder != 'categorical':
        raise ValueError("The masked action objective requires a categorical decoder")
    if type(span) is not int or span < 1:
        raise ValueError("masked_action_span must be a positive integer")


def sample_action_mask(slots, filled, span, generator):
    """Uniformly draw one contiguous window per episode containing observed actions."""
    if type(span) is not int or span < 1:
        raise ValueError("masked_action_span must be a positive integer")
    active = slots & filled.unsqueeze(-1)
    observed = active.any(-1).cpu()
    counts = observed.sum(1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("Each trajectory needs at least one observed teammate action")
    # Short episodes retain at least one visible action-bearing step when possible.
    width = (counts - 1).clamp(min=1, max=span)
    times = torch.arange(observed.size(1), device='cpu').unsqueeze(0)
    ends = times + width
    prefix = F.pad(observed.long().cumsum(1), (1, 0))
    observed_in_window = prefix.gather(1, ends.clamp_max(observed.size(1))) - prefix[:, :-1]
    candidates = (ends <= filled.sum(1, keepdim=True).cpu()) & (observed_in_window > 0)
    starts = torch.multinomial(candidates.float(), 1, generator=generator)
    window = (times >= starts) & (times < starts + width)
    return active & window.to(active.device).unsqueeze(-1)


class TeammateCRP:
    """Sequential MAP assignments to an unbounded set of teammate clusters.

    The small strategy probes are kept on CPU. Each model update trains on a
    minibatch from the union of previously seen and current group trajectories.
    """

    model_type = BehaviorModel

    def __init__(self, state_dim, n_actions, n_teammates, *, alpha, device='cpu', lr=.0005,
                 minibatch_size=32, decoder='paper', masked_action_weight=0., masked_action_span=3,
                 mask_seed=0):
        if not (alpha > 0 and isfinite(alpha) and lr > 0 and minibatch_size > 0):
            raise ValueError("Invalid CRP concentration or optimizer settings")
        self.alpha = alpha
        self.device = torch.device(device)
        self.minibatch_size = minibatch_size
        validate_masked_action_config(decoder, masked_action_weight, masked_action_span)
        self.masked_action_weight = masked_action_weight
        self.masked_action_span = masked_action_span
        self.mask_generator = torch.Generator(device='cpu').manual_seed(mask_seed)
        # Existing paper-model subclasses can keep their three-argument constructor.
        decoder_args = {} if decoder == 'paper' else {'decoder': decoder}
        self.model = self.model_type(state_dim, n_actions, n_teammates, **decoder_args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.assignments = {}
        self.counts = []
        self.centers = []
        self._probes = []
        self.last_model_loss = None
        self.last_reconstruction_loss = None
        self.last_masked_action_loss = None
        self.last_assignment_log_scores = None

    def _unpack(self, batch):
        valid = batch['filled'].bool().cpu()
        if not valid.numel() or not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("CRP needs a nonempty prefix of valid transitions for every probe")
        state = batch['state'].detach().cpu()
        actions = batch['teammate_actions'].detach().cpu()
        slots = batch['teammate_mask'].detach().bool().cpu()
        if (state.shape[:2] != (valid.size(0), valid.size(1) + 1)
                or actions.shape != (*valid.shape, self.model.n_teammates) or slots.shape != actions.shape):
            raise ValueError("Inconsistent state, action or mask shapes in CRP probes")
        active = slots & valid.unsqueeze(-1)
        if ((actions[active] < 0) | (actions[active] >= self.model.n_actions)).any():
            raise ValueError("A present teammate action is out of range")
        if not active.any((1, 2)).all():
            raise ValueError("Each CRP trajectory needs at least one observed teammate action")
        # Clone compact slices to avoid retaining padded batches or other replay data.
        return [(state[i, :length + 1].clone(), actions[i, :length].clone(),
                 slots[i, :length].clone(), valid[i, :length].clone())
                for i, length in enumerate(valid.sum(1).tolist())]

    def _model_loss(self, log_likelihood):
        # Eq. 4 places the trajectory expectation INSIDE the logarithm.
        return log(len(log_likelihood)) - torch.logsumexp(log_likelihood, dim=0)

    def behavior_loss(self, states, actions, slots, filled, behavior):
        if self.model.decoder == 'categorical':
            return self.model.episode_action_nll(states, actions, slots, filled, behavior).mean()
        return self._model_loss(self.model.trajectory_log_prob(states, actions, slots, filled, behavior))

    def _fit(self, probes, steps):
        self.model.train()
        for _ in range(steps):
            indices = torch.randperm(len(probes), device='cpu')[:self.minibatch_size].tolist()
            sample = _collate([probes[i] for i in indices], self.device)
            embedding = self.model.encode(*sample)
            reconstruction = self.behavior_loss(*sample, embedding)
            loss = reconstruction
            masked_nll = None
            if self.masked_action_weight:
                states, actions, slots, filled = sample
                hidden = sample_action_mask(slots, filled, self.masked_action_span, self.mask_generator)
                context = self.model.encode(*sample, action_mask=hidden)
                masked_nll = self.model.episode_action_nll(states, actions, hidden, filled, context).mean()
                loss = reconstruction + self.masked_action_weight * masked_nll
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10., error_if_nonfinite=True)
            self.optimizer.step()
        self.last_model_loss = loss.detach().item()
        self.last_reconstruction_loss = reconstruction.detach().item()
        self.last_masked_action_loss = None if masked_nll is None else masked_nll.detach().item()

    @torch.no_grad()
    def _assign(self, group):
        self.model.eval()
        # The current group may itself exceed one training minibatch.
        batches = [group[start:start + self.minibatch_size]
                   for start in range(0, len(group), self.minibatch_size)]
        vector = sum(self.model.encode(*_collate(probes, self.device)).sum(0)
                     for probes in batches) / len(group)
        candidates = [(n * center + vector) / (n + 1) for n, center in zip(self.counts, self.centers)]
        candidates.append(vector)
        # The common CRP normalizer K+alpha cancels in the MAP. Independent
        # trajectories from the new group contribute additive log evidence.
        scores = vector.new_tensor([*self.counts, self.alpha], dtype=torch.float64).log()
        for probes in batches:
            sample = _collate(probes, self.device)
            for i, center in enumerate(candidates):
                evidence = self.model.trajectory_log_prob(*sample, center.expand(len(probes), -1))
                scores[i] += evidence.double().sum()
        if not scores.isfinite().all():
            raise FloatingPointError("Nonfinite predictive likelihood in CRP")
        self.last_assignment_log_scores = scores.detach()
        label = scores.argmax().item()
        return label, candidates[label].detach()

    def add_group(self, name, batch, *, model_steps=1):
        if name in self.assignments:
            raise ValueError(f"Teammate group already assigned: {name}")
        if not isinstance(model_steps, int) or model_steps < 1:
            raise ValueError("At least one behavior-model update is required before assignment")
        group = self._unpack(batch)
        self._fit(self._probes + group, model_steps)
        label, center = self._assign(group)
        if label == len(self.counts):
            self.counts.append(1)
            self.centers.append(center)
        else:
            self.counts[label] += 1
            self.centers[label] = center
        self._probes.extend(group)
        self.assignments[name] = label
        return label

    def state_dict(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    assignments=dict(self.assignments), counts=list(self.counts), centers=list(self.centers),
                    probes=list(self._probes), alpha=self.alpha, minibatch_size=self.minibatch_size,
                    decoder=self.model.decoder,
                    masked_action_weight=self.masked_action_weight, masked_action_span=self.masked_action_span,
                    mask_generator_state=self.mask_generator.get_state(),
                    last_model_loss=self.last_model_loss,
                    last_reconstruction_loss=self.last_reconstruction_loss,
                    last_masked_action_loss=self.last_masked_action_loss,
                    last_assignment_log_scores=self.last_assignment_log_scores)

    def load_state_dict(self, saved):
        if saved.get('decoder', 'paper') != self.model.decoder:
            raise ValueError("Saved CRP decoder does not match the model")
        weight, span = saved.get('masked_action_weight', 0.), saved.get('masked_action_span', 3)
        validate_masked_action_config(self.model.decoder, weight, span)
        if weight and 'mask_generator_state' not in saved:
            raise ValueError("Missing mask generator state for auxiliary training")
        self.model.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.alpha, self.minibatch_size = saved['alpha'], saved['minibatch_size']
        self.masked_action_weight, self.masked_action_span = weight, span
        if 'mask_generator_state' in saved:
            self.mask_generator.set_state(saved['mask_generator_state'].cpu())
        self.assignments, self.counts = dict(saved['assignments']), list(saved['counts'])
        self.centers = [center.to(self.device) for center in saved['centers']]
        self._probes = [tuple(field.cpu() for field in probe) for probe in saved['probes']]
        self.last_model_loss = saved['last_model_loss']
        self.last_reconstruction_loss = saved.get('last_reconstruction_loss', self.last_model_loss)
        self.last_masked_action_loss = saved.get('last_masked_action_loss')
        self.last_assignment_log_scores = saved.get('last_assignment_log_scores')
