"""Small PD target utilities independent of any simulator API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def compute_position_targets(
    normalized_action: Sequence[float] | np.ndarray,
    joint_names: Sequence[str],
    default_joint_pos: Mapping[str, float],
    action_scale: Mapping[str, float] | float,
    joint_limits: Mapping[str, tuple[float, float]] | None = None,
    current_joint_pos: Mapping[str, float] | None = None,
    relative_to_current: bool = False,
) -> dict[str, float]:
    """Convert normalized actions into target joint positions.

    Actions are clamped to [-1, 1], scaled per joint, added to either default
    or current joint positions, then clipped to configured joint limits.
    """
    action = np.asarray(normalized_action, dtype=float)
    if action.shape != (len(joint_names),):
        raise ValueError(f"Expected action shape {(len(joint_names),)}, got {action.shape}")

    clipped = np.clip(action, -1.0, 1.0)
    limits = joint_limits or {}
    targets: dict[str, float] = {}

    for i, name in enumerate(joint_names):
        base_map = current_joint_pos if relative_to_current else default_joint_pos
        if base_map is None or name not in base_map:
            raise KeyError(f"Missing base joint position for {name}")
        scale = _scale_for_joint(action_scale, name)
        target = float(base_map[name]) + float(clipped[i]) * scale
        if name in limits:
            lo, hi = limits[name]
            target = float(np.clip(target, lo, hi))
        targets[name] = target

    return targets


def _scale_for_joint(action_scale: Mapping[str, float] | float, name: str) -> float:
    if isinstance(action_scale, Mapping):
        if name in action_scale:
            return float(action_scale[name])
        return float(action_scale.get("default", 0.0))
    return float(action_scale)


def self_test() -> None:
    joints = ["j0", "j1"]
    targets = compute_position_targets(
        normalized_action=[2.0, -0.5],
        joint_names=joints,
        default_joint_pos={"j0": 0.0, "j1": 1.0},
        action_scale={"default": 0.2},
        joint_limits={"j0": (-0.1, 0.1), "j1": (0.0, 2.0)},
    )
    assert targets == {"j0": 0.1, "j1": 0.9}


if __name__ == "__main__":
    self_test()
    print("pd_controller self_test passed")
