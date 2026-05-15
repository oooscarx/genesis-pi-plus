#!/usr/bin/env python3
"""Smoke test scaffold: hold pi_plus at the default joint pose with PD targets."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.config import load_config
from genesis_pi_plus.genesis_adapter import add_ground, create_scene, init_genesis, load_pi_plus, step_scene
from genesis_pi_plus.pd_controller import compute_position_targets
from genesis_pi_plus.pi_plus_model import PiPlusModelInfo


def main() -> None:
    cfg = load_config(ROOT / "configs/pi_plus_genesis.yaml")
    cfg.setdefault("sim", {})["headless"] = True
    info = PiPlusModelInfo.from_config(cfg)
    if not info.joint_names:
        raise SystemExit("robot.joint_names is empty; fill configs/pi_plus_genesis.yaml first.")

    init_genesis(headless=True, backend=cfg.get("sim", {}).get("backend"))
    scene = create_scene(cfg)
    add_ground(scene, cfg)
    robot = load_pi_plus(scene, cfg)

    zero_action = np.zeros(len(info.joint_names), dtype=float)
    targets = compute_position_targets(
        zero_action,
        joint_names=info.joint_names,
        default_joint_pos=info.default_joint_pos,
        action_scale=cfg["robot"].get("action_scale", {"default": 0.0}),
        joint_limits=info.joint_limits,
    )
    apply_joint_targets(robot, targets)

    sim_dt = float(cfg.get("sim", {}).get("sim_dt") or 0.002)
    seconds = float(cfg.get("test", {}).get("stand_seconds") or 5.0)
    steps = max(1, int(seconds / sim_dt))
    step_scene(scene, steps)

    print("PD stand smoke test finished.")
    print("TODO: read base height/roll/pitch from Genesis and evaluate fall threshold.")


def apply_joint_targets(robot, targets: dict[str, float]) -> None:
    """Apply targets using the installed Genesis API once confirmed."""
    candidate_methods = ("set_dofs_position", "control_dofs_position", "set_joint_positions")
    for method in candidate_methods:
        if hasattr(robot, method):
            print(f"TODO: map named targets to Genesis dof indices before calling robot.{method}.")
            return
    raise RuntimeError("TODO: Genesis joint target API is not verified for pi_plus.")


if __name__ == "__main__":
    main()
