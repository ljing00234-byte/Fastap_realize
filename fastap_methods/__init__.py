"""Training methods selected independently of game and actor configuration."""

from .fastap_training import train_fastap
from .qmix import train_qmix

TRAINERS = {"qmix": train_qmix, "fastap": train_fastap}

__all__ = ["TRAINERS", "train_fastap", "train_qmix"]
