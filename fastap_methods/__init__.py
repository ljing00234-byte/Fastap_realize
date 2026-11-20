"""Training methods selected independently of game and actor configuration."""

from .qmix import train_qmix

TRAINERS = {"qmix": train_qmix}

__all__ = ["TRAINERS", "train_qmix"]
