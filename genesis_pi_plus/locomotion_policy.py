"""pi_plus locomotion policy adapter used as a residual-control baseline."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import types

import torch
from torch import nn

from .assets import ensure_exists, resolve_path
from .math_utils import quat_rotate_inverse_wxyz_torch


MUJOCO_TO_ISAAC_IDX = torch.tensor(
    [0, 6, 10, 16, 1, 7, 11, 17, 2, 8, 12, 18, 3, 9, 13, 19, 4, 14, 5, 15],
    dtype=torch.long,
)
ISAAC_TO_MUJOCO_IDX = torch.tensor(
    [0, 4, 8, 12, 16, 18, 1, 5, 9, 13, 2, 6, 10, 14, 17, 19, 3, 7, 11, 15],
    dtype=torch.long,
)


def resolve_policy_device(policy_device: str, backend_name: str | None = None) -> torch.device:
    """Choose the inference device for the frozen locomotion policy."""
    if policy_device != "auto":
        return torch.device(policy_device)
    backend = (backend_name or "").lower()
    if backend == "metal" and torch.backends.mps.is_available():
        return torch.device("mps")
    if backend in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_locomotion_policy(policy_path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Load model_40000.pt or an exported TorchScript locomotion actor."""
    path = ensure_exists(resolve_path(str(policy_path)))
    try:
        return torch.jit.load(str(path), map_location=device).eval()
    except RuntimeError:
        pass

    install_amp_tk_import_shims()
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported locomotion policy file: {path}")
    state_dict = checkpoint["model_state_dict"]
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in state_dict.items()
        if key.startswith("actor.")
    }
    actor = nn.Sequential(
        nn.Linear(345, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 20),
    )
    actor.load_state_dict(actor_state)
    return actor.to(device).eval()


def install_amp_tk_import_shims() -> None:
    """Allow torch.load to unpickle AMP_TK checkpoint metadata without extra deps."""
    root = Path(__file__).resolve().parents[1]
    amp_root = (root / "../AMP_TK").resolve()
    for entry in (amp_root / "rsl_rl", amp_root):
        entry_str = str(entry)
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)
    if "pybullet_utils" not in sys.modules:
        pybullet_utils = types.ModuleType("pybullet_utils")
        transformations = types.ModuleType("transformations")
        pybullet_utils.transformations = transformations
        sys.modules["pybullet_utils"] = pybullet_utils
        sys.modules["pybullet_utils.transformations"] = transformations
    if "git" not in sys.modules:
        sys.modules["git"] = types.ModuleType("git")
    _install_normalizer_compat()


def _install_normalizer_compat() -> None:
    """Patch pip rsl-rl with old AMP_TK class names when it is already imported."""

    class RunningMeanStd:
        pass

    class Normalizer(RunningMeanStd):
        pass

    for module_name in ("rsl_rl.utils.utils", "rsl_rl.utils"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if not hasattr(module, "RunningMeanStd"):
            setattr(module, "RunningMeanStd", RunningMeanStd)
        if not hasattr(module, "Normalizer"):
            setattr(module, "Normalizer", Normalizer)


@dataclass
class LocomotionBaseline:
    """Frozen 69x5-history pi_plus locomotion actor.

    Inputs and output order mirror ``sim2sim_pi_plus.py``:
    q/dq are in configured MJCF joint order, policy action is Isaac order, and
    joint targets are returned in configured MJCF joint order.
    """

    policy: nn.Module
    device: torch.device
    action_scale: float = 0.25
    history_len: int = 5

    def __post_init__(self) -> None:
        self.obs_history: torch.Tensor | None = None
        self.last_action_isaac: torch.Tensor | None = None
        self._mujoco_to_isaac = MUJOCO_TO_ISAAC_IDX.to(self.device)
        self._isaac_to_mujoco = ISAAC_TO_MUJOCO_IDX.to(self.device)

    def reset(self, num_envs: int, env_ids: torch.Tensor | None = None) -> None:
        if self.obs_history is None or self.obs_history.shape[0] != num_envs:
            self.obs_history = torch.zeros(num_envs, 69 * self.history_len, dtype=torch.float32, device=self.device)
            self.last_action_isaac = torch.zeros(num_envs, 20, dtype=torch.float32, device=self.device)
            return
        if env_ids is None:
            self.obs_history.zero_()
            self.last_action_isaac.zero_()
        else:
            env_ids = env_ids.to(self.device)
            self.obs_history[env_ids] = 0.0
            self.last_action_isaac[env_ids] = 0.0

    @torch.no_grad()
    def target(
        self,
        *,
        q_mujoco: torch.Tensor,
        dq_mujoco: torch.Tensor,
        default_pos: torch.Tensor,
        base_quat_wxyz: torch.Tensor,
        base_ang_vel_world: torch.Tensor,
        command: torch.Tensor,
    ) -> torch.Tensor:
        if self.obs_history is None or self.last_action_isaac is None:
            self.reset(q_mujoco.shape[0])

        q_policy = q_mujoco.to(self.device)
        dq_policy = dq_mujoco.to(self.device)
        default_policy = default_pos.to(self.device)
        quat_policy = base_quat_wxyz.to(self.device)
        ang_policy = base_ang_vel_world.to(self.device)
        command_policy = command.to(self.device)

        obs = torch.zeros(q_policy.shape[0], 69, dtype=torch.float32, device=self.device)
        obs[:, 0:3] = quat_rotate_inverse_wxyz_torch(quat_policy, ang_policy)
        gravity = torch.zeros(q_policy.shape[0], 3, dtype=torch.float32, device=self.device)
        gravity[:, 2] = -1.0
        obs[:, 3:6] = quat_rotate_inverse_wxyz_torch(quat_policy, gravity)
        obs[:, 6:9] = command_policy
        obs[:, 9:29] = (q_policy - default_policy)[:, self._mujoco_to_isaac]
        obs[:, 29:49] = dq_policy[:, self._mujoco_to_isaac]
        obs[:, 49:69] = torch.clamp(self.last_action_isaac, -100.0, 100.0)

        self.obs_history = torch.roll(self.obs_history, shifts=-69, dims=1)
        self.obs_history[:, -69:] = obs
        self.obs_history = torch.clamp(self.obs_history, -100.0, 100.0)

        action = self.policy(self.obs_history)
        action = torch.clamp(action[:, :20].float(), -100.0, 100.0)
        self.last_action_isaac = action
        target_policy = action[:, self._isaac_to_mujoco] * float(self.action_scale) + default_policy
        return target_policy.to(q_mujoco.device)
