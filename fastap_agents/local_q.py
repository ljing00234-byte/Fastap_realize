"""Local Q policy compatible with the pinned EPyMARL BasicMAC checkpoint."""

from dataclasses import asdict, dataclass
from functools import partial
import json
from operator import index
from pathlib import Path
import sys
from typing import ClassVar

import torch


def _add_epymarl_src():
    # EPyMARL uses absolute imports rooted at its src directory.
    src = str(Path(__file__).resolve().parents[1] / "third_party" / "epymarl" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


@dataclass(frozen=True)
class AgentSpec:
    """Inputs and architecture shared by training and decentralized execution."""

    FORMAT: ClassVar[str] = "epymarl-shared-q-v1"
    scene_key: str
    control: str
    n_agents: int
    obs_shape: int
    n_actions: int
    hidden_dim: int
    use_rnn: bool
    obs_last_action: bool
    obs_agent_id: bool

    def __post_init__(self):
        if not self.scene_key or self.control not in ("ego", "team"):
            raise ValueError("An agent needs a scene key and control mode 'ego' or 'team'")
        for key in ("n_agents", "obs_shape", "n_actions", "hidden_dim"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("use_rnn", "obs_last_action", "obs_agent_id"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")

    @classmethod
    def from_epymarl(cls, env_info, config, *, scene_key, control):
        if (config["mac"], config["agent"], config["agent_output_type"]) != ("basic_mac", "rnn", "q"):
            raise ValueError("Expected the shared BasicMAC local Q network (agent='rnn')")
        return cls(scene_key=scene_key, control=control, n_agents=env_info["n_agents"],
                   obs_shape=env_info["obs_shape"], n_actions=env_info["n_actions"],
                   hidden_dim=config["hidden_dim"], use_rnn=config["use_rnn"],
                   obs_last_action=config["obs_last_action"], obs_agent_id=config["obs_agent_id"])

    def save(self, checkpoint_dir):
        """Store the input layout beside the training controller's agent.th weights."""
        contents = dict(format=self.FORMAT, **asdict(self))
        (Path(checkpoint_dir) / "agent.json").write_text(
            json.dumps(contents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    @classmethod
    def load(cls, checkpoint_dir):
        contents = json.loads((Path(checkpoint_dir) / "agent.json").read_text(encoding="utf-8"))
        if contents.pop("format", None) != cls.FORMAT:
            raise ValueError("Unsupported agent checkpoint format")
        return cls(**contents)

    def build_network(self):
        _add_epymarl_src()
        from modules.agents.rnn_agent import RNNAgent

        size = self.obs_shape
        if self.obs_last_action:
            size += self.n_actions
        if self.obs_agent_id:
            size += self.n_agents
        return RNNAgent(size, self)


class FrozenAgent:
    """Shared evaluation weights; each original team slot gets its own history."""

    def __init__(self, spec, model):
        self.spec = spec
        self.model = model.eval().requires_grad_(False)

    @classmethod
    def load(cls, checkpoint_dir, *, spec=None, device="cpu"):
        """Older agent.th files can be loaded with their original explicit spec."""
        if spec is None:
            spec = AgentSpec.load(checkpoint_dir)
        weights = torch.load(Path(checkpoint_dir) / "agent.th", map_location="cpu", weights_only=True)
        # Constructing a temporary network must not change the training RNG.
        with torch.random.fork_rng(devices=[]):
            model = spec.build_network()
        model.load_state_dict(weights, strict=True)
        return cls(spec, model.to(device))

    def policy_factory(self, slot):
        """Create a fresh local policy for this zero-based slot of the training roster."""
        slot = index(slot)
        if not 0 <= slot < self.spec.n_agents:
            raise ValueError(f"Slot {slot} is outside the original {self.spec.n_agents}-agent roster")
        return partial(_SlotPolicy, self.spec, self.model, slot)


class _SlotPolicy:
    def __init__(self, spec, model, slot):
        self.spec = spec
        self.model = model
        self.hidden = model.init_hidden()
        self.last_action = self.hidden.new_zeros(spec.n_actions)
        self.role = self.hidden.new_zeros(spec.n_agents)
        self.role[slot] = 1

    @torch.inference_mode()
    def __call__(self, obs, available, rng):
        """Choose a greedy legal action using this slot's local information."""
        local_obs = torch.as_tensor(obs, dtype=self.hidden.dtype, device=self.hidden.device)
        mask = torch.as_tensor(available, device=self.hidden.device) != 0
        if local_obs.shape != (self.spec.obs_shape,):
            raise ValueError(f"Expected a local observation with {self.spec.obs_shape} features")
        if mask.shape != (self.spec.n_actions,) or not mask.any():
            raise ValueError(f"Expected a {self.spec.n_actions}-action mask with at least one legal action")
        features = [local_obs]
        if self.spec.obs_last_action:
            features.append(self.last_action)
        if self.spec.obs_agent_id:
            features.append(self.role)
        q_values, self.hidden = self.model(torch.cat(features).unsqueeze(0), self.hidden)
        action = int(q_values[0].masked_fill(~mask, -torch.inf).argmax().item())
        if self.spec.obs_last_action:
            self.last_action.zero_()
            self.last_action[action] = 1
        return action
