#!/usr/bin/env python3
"""Run the exported pi_plus locomotion policy in Genesis.

This is a sim2sim smoke test, not training. It mirrors the observation/action
layout used by ../sim2sim_pi_plus.py and maps all Genesis DOFs by joint name.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.assets import ensure_exists
from genesis_pi_plus.config import load_config
from genesis_pi_plus.genesis_adapter import add_ground, create_scene, init_genesis, load_pi_plus
from genesis_pi_plus.pi_plus_model import PiPlusModelInfo

MUJOCO_TO_ISAAC_IDX = [0, 6, 10, 16, 1, 7, 11, 17, 2, 8, 12, 18, 3, 9, 13, 19, 4, 14, 5, 15]
ISAAC_TO_MUJOCO_IDX = [0, 4, 8, 12, 16, 18, 1, 5, 9, 13, 2, 6, 10, 14, 17, 19, 3, 7, 11, 15]
BASE_LINK_LOCAL_IDX = 1


def main() -> None:
    args = parse_args()
    cfg = load_config(ROOT / "configs/pi_plus_genesis.yaml")
    cfg.setdefault("sim", {})["headless"] = args.headless

    info = PiPlusModelInfo.from_config(cfg)
    policy_path = ensure_exists(args.policy)
    policy = load_policy(policy_path)
    policy.eval()

    backend_name = args.backend or os.environ.get("GENESIS_BACKEND") or cfg.get("sim", {}).get("backend")
    init_genesis(headless=args.headless, backend=backend_name)
    policy_device = resolve_policy_device(args.policy_device, backend_name)
    policy = policy.to(policy_device)
    print(f"Policy inference device: {policy_device}")
    scene = create_scene(cfg)
    add_ground(scene, cfg)
    robot = load_pi_plus(scene, cfg)
    scene.build()

    joint_names = info.joint_names
    if len(joint_names) != 20:
        raise RuntimeError(f"Expected 20 pi_plus action joints, got {len(joint_names)}.")

    dof_idx = dof_indices_by_joint_name(robot, joint_names)
    default_pos = np.array([info.default_joint_pos[name] for name in joint_names], dtype=np.float32)
    kp = np.array([info.pd_kp[name] for name in joint_names], dtype=np.float32)
    kd = np.array([info.pd_kd[name] for name in joint_names], dtype=np.float32)
    action_scale = float(cfg["robot"].get("action_scale", {}).get("default", 0.25))

    set_initial_pose(robot, dof_idx, default_pos)
    obs_history = np.zeros(69 * 5, dtype=np.float32)
    action = np.zeros(20, dtype=np.float32)
    command = np.array([args.x_vel, args.y_vel, args.yaw_vel], dtype=np.float32)

    sim_dt = float(cfg.get("sim", {}).get("sim_dt") or 0.002)
    decimation = max(1, round(float(cfg.get("sim", {}).get("control_dt") or 0.02) / sim_dt))
    total_control_steps = max(1, int(args.duration / (sim_dt * decimation)))
    max_torque = np.array([20.0] * 6 + [10.0] * 4 + [20.0] * 6 + [10.0] * 4, dtype=np.float32)

    print(f"Policy: {policy_path}")
    print(f"Command: vx={args.x_vel:.2f}, vy={args.y_vel:.2f}, wz={args.yaw_vel:.2f}")
    print(f"Running {args.duration:.2f}s, sim_dt={sim_dt}, decimation={decimation}")

    for control_step in range(total_control_steps):
        q = to_numpy(robot.get_dofs_position(dof_idx)).astype(np.float32)
        dq = to_numpy(robot.get_dofs_velocity(dof_idx)).astype(np.float32)
        obs_history = update_obs_history(robot, q, dq, default_pos, action, command, obs_history)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs_history, dtype=torch.float32, device=policy_device).unsqueeze(0)
            action = policy(obs_tensor).detach().cpu().numpy()[0, :20].astype(np.float32)
        action = np.clip(action, -100.0, 100.0)

        target_mujoco_order = action[ISAAC_TO_MUJOCO_IDX] * action_scale + default_pos
        for _ in range(decimation):
            q = to_numpy(robot.get_dofs_position(dof_idx)).astype(np.float32)
            dq = to_numpy(robot.get_dofs_velocity(dof_idx)).astype(np.float32)
            tau = (target_mujoco_order - q) * kp - dq * kd
            tau = np.clip(tau, -max_torque, max_torque)
            robot.control_dofs_force(tau, dof_idx)
            scene.step(update_visualizer=not args.headless, refresh_visualizer=not args.headless)

        if control_step % args.log_every == 0:
            pos, quat = get_base_pose(robot)
            roll, pitch, yaw = quat_wxyz_to_rpy(quat)
            print(
                f"t={control_step * decimation * sim_dt:6.3f}s "
                f"base_z={pos[2]: .3f} roll={roll: .3f} pitch={pitch: .3f} yaw={yaw: .3f} "
                f"action_norm={np.linalg.norm(action): .3f}"
            )

    print("Genesis pi_plus policy rollout finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model_40000.pt on pi_plus in Genesis.")
    parser.add_argument(
        "--policy",
        default="policies/model_40000.pt",
        help="TorchScript or AMP_TK checkpoint policy path, relative to genesis_pi_plus/ by default.",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--x-vel", type=float, default=0.8)
    parser.add_argument("--y-vel", type=float, default=0.0)
    parser.add_argument("--yaw-vel", type=float, default=0.0)
    parser.add_argument("--backend", default=None, help="Genesis backend override: cpu, metal, cuda, gpu.")
    parser.add_argument("--policy-device", default="auto", help="Policy inference device: auto, cpu, mps, cuda.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def resolve_policy_device(policy_device: str, backend_name: str | None) -> torch.device:
    if policy_device != "auto":
        return torch.device(policy_device)
    backend = (backend_name or "").lower()
    if backend == "metal" and torch.backends.mps.is_available():
        return torch.device("mps")
    if backend in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_policy(policy_path: Path) -> torch.nn.Module:
    """Load either an exported TorchScript policy or an AMP_TK checkpoint."""
    try:
        return torch.jit.load(str(policy_path), map_location="cpu")
    except RuntimeError as exc:
        print(f"TorchScript load failed, trying AMP_TK checkpoint actor: {exc}")

    install_amp_tk_import_shims()
    checkpoint = torch.load(str(policy_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported policy file format: {policy_path}")
    state_dict = checkpoint["model_state_dict"]
    actor_state = {key.removeprefix("actor."): value for key, value in state_dict.items() if key.startswith("actor.")}
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
    print("Loaded AMP_TK checkpoint actor MLP [345, 512, 256, 128, 20].")
    return actor


def install_amp_tk_import_shims() -> None:
    """Allow torch.load to unpickle AMP_TK checkpoint metadata without new deps."""
    amp_root = ROOT / "../AMP_TK"
    sys.path.insert(0, str((amp_root / "rsl_rl").resolve()))
    sys.path.insert(0, str(amp_root.resolve()))
    if "pybullet_utils" not in sys.modules:
        pybullet_utils = types.ModuleType("pybullet_utils")
        transformations = types.ModuleType("transformations")
        pybullet_utils.transformations = transformations
        sys.modules["pybullet_utils"] = pybullet_utils
        sys.modules["pybullet_utils.transformations"] = transformations
    if "git" not in sys.modules:
        sys.modules["git"] = types.ModuleType("git")


def dof_indices_by_joint_name(robot, joint_names: list[str]) -> list[int]:
    by_name = {joint.name: joint for joint in robot.joints}
    missing = [name for name in joint_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Genesis robot is missing configured joints: {missing}")
    indices = []
    for name in joint_names:
        joint = by_name[name]
        if len(joint.dofs_idx_local) != 1:
            raise RuntimeError(f"Expected one DOF for {name}, got {joint.dofs_idx_local}")
        indices.append(int(joint.dofs_idx_local[0]))
    print("Genesis DOF mapping:")
    for name, idx in zip(joint_names, indices, strict=True):
        print(f"  {idx:2d}: {name}")
    return indices


def set_initial_pose(robot, dof_idx: list[int], default_pos: np.ndarray) -> None:
    robot.set_dofs_position(default_pos, dof_idx, zero_velocity=True)
    robot.zero_all_dofs_velocity()


def update_obs_history(
    robot,
    q_mujoco_order: np.ndarray,
    dq_mujoco_order: np.ndarray,
    default_pos: np.ndarray,
    action_isaac_order: np.ndarray,
    command: np.ndarray,
    obs_history: np.ndarray,
) -> np.ndarray:
    obs = np.zeros(69, dtype=np.float32)
    _, quat_wxyz = get_base_pose(robot)
    quat_wxyz = quat_wxyz.astype(np.float64)
    ang_vel_world = to_numpy(robot.get_links_ang([BASE_LINK_LOCAL_IDX])).reshape(-1)[:3].astype(np.float64)

    obs[0:3] = quat_rotate_inverse_wxyz(quat_wxyz, ang_vel_world).astype(np.float32)
    obs[3:6] = quat_rotate_inverse_wxyz(quat_wxyz, np.array([0.0, 0.0, -1.0])).astype(np.float32)
    obs[6:9] = command
    obs[9:29] = (q_mujoco_order - default_pos)[MUJOCO_TO_ISAAC_IDX]
    obs[29:49] = dq_mujoco_order[MUJOCO_TO_ISAAC_IDX]
    obs[49:69] = np.clip(action_isaac_order, -100.0, 100.0)

    obs_history = np.roll(obs_history, shift=-69)
    obs_history[-69:] = obs
    return np.clip(obs_history, -100.0, 100.0)


def get_base_pose(robot) -> tuple[np.ndarray, np.ndarray]:
    pos = to_numpy(robot.get_links_pos([BASE_LINK_LOCAL_IDX])).reshape(-1, 3)[0]
    quat = to_numpy(robot.get_links_quat([BASE_LINK_LOCAL_IDX])).reshape(-1, 4)[0]
    return pos, quat


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    q_vec = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v - 2.0 * (w * uv + uuv)


def quat_wxyz_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    main()
