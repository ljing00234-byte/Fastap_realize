"""Batched soft collision forces for the MPE particle scenes."""

import torch

from .particle import _CONTACT_FORCE, _CONTACT_MARGIN


class ParticleContacts:
    def __init__(self, radii, movers):
        device = radii.device
        self.movers = movers
        self.touch_distance = radii[:movers, None] + radii[None, :]
        self.interact = ~torch.eye(movers, len(radii), dtype=torch.bool, device=device)
        # At exact coincidence the scalar pair loop pushes the lower-index body towards +x.
        first = torch.arange(movers, device=device)[:, None]
        second = torch.arange(len(radii), device=device)
        signs = torch.where(first < second, 1, -1)
        self.coincident_normal = torch.stack((signs, torch.zeros_like(signs)), dim=-1)

    def forces(self, positions):
        delta = positions[:, :self.movers, None, :] - positions[:, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        normal = delta / distance.masked_fill(distance == 0, 1).unsqueeze(-1)
        normal = torch.where((distance == 0).unsqueeze(-1), self.coincident_normal, normal)
        penetration = torch.logaddexp(distance.new_zeros(()), (self.touch_distance - distance) / _CONTACT_MARGIN)
        return ((_CONTACT_FORCE * _CONTACT_MARGIN * penetration * self.interact).unsqueeze(-1) * normal).sum(2)
