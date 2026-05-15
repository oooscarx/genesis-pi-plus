"""Genesis vectorized environment for pi_plus residual kick training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config
from .genesis_adapter import add_ball, add_ground, create_scene, init_genesis, load_pi_plus
from .kick_rewards import KickRewardScales, compute_kick_rewards
from .math_utils import normalize_xy, quat_rotate_inverse_wxyz_torch, quat_wxyz_to_rpy_torch, to_numpy
from .pi_plus_model import PiPlusModelInfo
from .safety import ResidualSafetyFilter, SafetyLimits


BASE_LINK_LOCAL_IDX_FALLBACK = 1


@dataclass(frozen=True)
class KickEnvPaths:
    robot_cfg: Path
    train_cfg: Path
    reward_cfg: Path
    domain_rand_cfg: Path


class PiPlusKickEnv:
    """Residual joint target kick environment with an rsl-rl VecEnv-like API."""

    num_actions = 20

    def __init__(
        self,
        paths: KickEnvPaths,
        *,
        num_envs: int | None = None,
        device: str | torch.device | None = None,
        backend: str | None = None,
        headless: bool = True,
        build_scene: bool = True,
    ):
        self.robot_cfg = load_config(paths.robot_cfg)
        self.train_cfg = load_config(paths.train_cfg)
        self.reward_cfg = load_config(paths.reward_cfg)
        self.domain_rand_cfg = load_config(paths.domain_rand_cfg)
        self.cfg = {"robot": self.robot_cfg, "train": self.train_cfg, "reward": self.reward_cfg, "domain_randomization": self.domain_rand_cfg}

        self.num_envs = int(num_envs or self.train_cfg.get("num_envs", 1))
        self.device = torch.device(device or _resolve_torch_device(self.train_cfg.get("device", "cuda")))
        self.robot_cfg.setdefault("sim", {})["headless"] = headless
        self.control_dt = float(self.robot_cfg.get("sim", {}).get("control_dt") or 0.02)
        self.sim_dt = float(self.robot_cfg.get("sim", {}).get("sim_dt") or 0.002)
        self.decimation = max(1, round(self.control_dt / self.sim_dt))
        self.max_episode_length = int(float(self.train_cfg.get("episode_length_s", 2.5)) / self.control_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.model_info = PiPlusModelInfo.from_config(self.robot_cfg)
        self.joint_names = self.model_info.joint_names
        if len(self.joint_names) != self.num_actions:
            raise ValueError(f"Expected 20 pi_plus joints, got {len(self.joint_names)}")

        self.default_pos = torch.tensor([self.model_info.default_joint_pos[n] for n in self.joint_names], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.kp = torch.tensor([self.model_info.pd_kp[n] for n in self.joint_names], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.kd = torch.tensor([self.model_info.pd_kd[n] for n in self.joint_names], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.action_scale = self._make_action_scale().repeat(self.num_envs, 1)
        self.max_torque = self._make_max_torque().repeat(self.num_envs, 1)

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_delta = torch.zeros_like(self.actions)
        self.last_torque = torch.zeros_like(self.actions)
        self.ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_ball_pos = torch.zeros_like(self.ball_pos)
        self.target_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.desired_ball_speed = torch.zeros(self.num_envs, device=self.device)
        self.extras: dict[str, Any] = {"observations": {}, "log": {}}

        self.safety_filter = ResidualSafetyFilter(self._make_safety_limits(), self.action_scale)
        self.reward_scales = KickRewardScales(**self.reward_cfg["scales"])

        self.gs = None
        self.scene = None
        self.robot = None
        self.ball = None
        self.dof_idx: list[int] = list(range(self.num_actions))
        self.base_link_idx = BASE_LINK_LOCAL_IDX_FALLBACK
        self.foot_link_idx: list[int] = []
        if build_scene:
            self._build_genesis(backend, headless)
            self.reset()

    @property
    def num_obs(self) -> int:
        return 3 + 3 + 20 + 20 + 20 + 3 + 2 + 2 + 2

    @property
    def unwrapped(self) -> "PiPlusKickEnv":
        return self

    @property
    def step_dt(self) -> float:
        return self.control_dt

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs = self._compute_observations()
        self.extras["observations"] = {"critic": obs}
        return obs, self.extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(self.device).float()
        self.prev_actions = self.actions.clone()
        self.prev_ball_pos = self.ball_pos.clone()

        q, dq = self._get_joint_state()
        base_pos, base_quat = self._get_base_pose()
        base_rpy = quat_wxyz_to_rpy_torch(base_quat)
        kickable = self._is_kickable(self.ball_pos)
        safe_delta, unclamped_delta = self.safety_filter.filter_action(
            actions, self.prev_delta, dq, base_rpy, base_pos[:, 2], kickable
        )
        self.actions = safe_delta
        self.prev_delta = safe_delta

        target = self.default_pos + safe_delta
        for _ in range(self.decimation):
            q, dq = self._get_joint_state()
            torque = torch.clamp((target - q) * self.kp - dq * self.kd, -self.max_torque, self.max_torque)
            self._control_torque(torque)
            self.last_torque = torque
            if self.scene is not None:
                self.scene.step(update_visualizer=False, refresh_visualizer=False)

        self.episode_length_buf += 1
        self.ball_pos = self._get_ball_pos()
        base_pos, base_quat = self._get_base_pose()
        base_rpy = quat_wxyz_to_rpy_torch(base_quat)
        fallen = self._fallen(base_pos, base_rpy)
        timeout = self.episode_length_buf >= self.max_episode_length
        done = fallen | timeout
        contact = self._ball_foot_contact()

        rewards, terms = compute_kick_rewards(
            scales=self.reward_scales,
            ball_pos=self.ball_pos,
            prev_ball_pos=self.prev_ball_pos,
            target_pos=self.target_pos,
            desired_ball_speed=self.desired_ball_speed,
            base_rpy=base_rpy,
            base_height=base_pos[:, 2],
            action=safe_delta,
            prev_action=self.prev_actions,
            torque=self.last_torque,
            contact=contact,
            kickable=kickable,
            fallen=fallen,
            control_dt=self.control_dt,
            base_height_target=float(self.reward_cfg["thresholds"]["base_height_target"]),
            base_height_sigma=float(self.reward_cfg["thresholds"]["base_height_sigma"]),
        )

        if torch.any(done):
            self._reset_idx(torch.nonzero(done, as_tuple=False).flatten())

        obs = self._compute_observations()
        self.extras = {
            "observations": {"critic": obs},
            "time_outs": timeout,
            "log": {f"reward/{k}": v.mean().detach() for k, v in terms.items()},
        }
        self.extras["log"]["episode/success_proxy"] = (torch.linalg.norm(self.ball_pos[:, :2] - self.target_pos, dim=-1) < float(self.reward_cfg["thresholds"]["success_distance_m"])).float().mean()
        self.extras["log"]["episode/fall_rate"] = fallen.float().mean()
        return obs, rewards, done, self.extras

    def _build_genesis(self, backend: str | None, headless: bool) -> None:
        self.gs = init_genesis(headless=headless, backend=backend or self.robot_cfg.get("sim", {}).get("backend"))
        self.scene = create_scene(self.robot_cfg)
        add_ground(self.scene, self.robot_cfg)
        self.robot = load_pi_plus(self.scene, self.robot_cfg)
        self.ball = add_ball(self.scene, self.robot_cfg)
        env_spacing = tuple(self.train_cfg.get("env_spacing", [2.0, 2.0]))
        self.scene.build(n_envs=self.num_envs, env_spacing=env_spacing)
        self.dof_idx = self._dof_indices_by_joint_name()
        self.base_link_idx = self._base_link_idx()
        self.foot_link_idx = self._link_indices(self.model_info.foot_link_names or ["l_ankle_roll_link", "r_ankle_roll_link"])

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.episode_length_buf[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_delta[env_ids] = 0.0
        self.last_torque[env_ids] = 0.0
        self._sample_task(env_ids)
        if self.robot is not None:
            self._set_dofs_position(self.default_pos[env_ids], env_ids)
            self.robot.zero_all_dofs_velocity(envs_idx=env_ids)
        if self.ball is not None:
            self._set_entity_pos(self.ball, self.ball_pos[env_ids], env_ids)
        self.prev_ball_pos[env_ids] = self.ball_pos[env_ids]

    def _sample_task(self, env_ids: torch.Tensor) -> None:
        task = self.train_cfg["task"]
        n = env_ids.numel()
        ball = torch.tensor(task["ball_initial_pos"], dtype=torch.float32, device=self.device).repeat(n, 1)
        if self._curriculum_flag("randomize_ball"):
            xr = task["ball_pos_range"]["x"]
            yr = task["ball_pos_range"]["y"]
            ball[:, 0] = _rand_uniform(xr[0], xr[1], n, self.device)
            ball[:, 1] = _rand_uniform(yr[0], yr[1], n, self.device)
        self.ball_pos[env_ids] = ball

        if self._curriculum_flag("randomize_target"):
            ar = self._stage_value("target_angle_range_rad", task["target_angle_range_rad"])
            dr = task["target_distance_range"]
            angle = _rand_uniform(ar[0], ar[1], n, self.device)
            distance = _rand_uniform(dr[0], dr[1], n, self.device)
            target = torch.stack((torch.cos(angle) * distance, torch.sin(angle) * distance), dim=-1)
        else:
            target = torch.tensor(task["target_pos_robot"], dtype=torch.float32, device=self.device).repeat(n, 1)
        self.target_pos[env_ids] = target

        sr = task.get("desired_ball_speed_range", [task["desired_ball_speed"], task["desired_ball_speed"]])
        self.desired_ball_speed[env_ids] = _rand_uniform(sr[0], sr[1], n, self.device)

    def _compute_observations(self) -> torch.Tensor:
        q, dq = self._get_joint_state()
        base_pos, base_quat = self._get_base_pose()
        ang_vel = self._get_base_ang_vel()
        projected_gravity = quat_rotate_inverse_wxyz_torch(base_quat, torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1))
        ball_rel = self.ball_pos - base_pos
        ball_vel = (self.ball_pos - self.prev_ball_pos) / self.control_dt
        target_rel = self.target_pos - base_pos[:, :2]
        target_dir = normalize_xy(target_rel)
        target_dist = torch.linalg.norm(target_rel, dim=-1, keepdim=True)
        ang_vel = self._add_noise(ang_vel, "imu_std")
        q_err = self._add_noise(q - self.default_pos, "joint_pos_std")
        dq = self._add_noise(dq, "joint_vel_std")
        ball_rel = self._add_noise(ball_rel, "ball_pos_std")
        return torch.cat(
            [
                ang_vel,
                projected_gravity,
                q_err,
                dq,
                self.actions,
                ball_rel,
                ball_vel[:, :2],
                target_dir,
                target_dist,
                self.desired_ball_speed[:, None],
            ],
            dim=-1,
        )

    def _get_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.robot is None:
            return self.default_pos.clone(), torch.zeros_like(self.default_pos)
        q = _as_torch(self.robot.get_dofs_position(self.dof_idx), self.device)
        dq = _as_torch(self.robot.get_dofs_velocity(self.dof_idx), self.device)
        return q.reshape(self.num_envs, -1), dq.reshape(self.num_envs, -1)

    def _get_base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.robot is None:
            pos = torch.zeros(self.num_envs, 3, device=self.device)
            pos[:, 2] = 0.38
            quat = torch.zeros(self.num_envs, 4, device=self.device)
            quat[:, 0] = 1.0
            return pos, quat
        pos = _as_torch(self.robot.get_links_pos([self.base_link_idx]), self.device).reshape(self.num_envs, 3)
        quat = _as_torch(self.robot.get_links_quat([self.base_link_idx]), self.device).reshape(self.num_envs, 4)
        return pos, quat

    def _get_base_ang_vel(self) -> torch.Tensor:
        if self.robot is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return _as_torch(self.robot.get_links_ang([self.base_link_idx]), self.device).reshape(self.num_envs, 3)

    def _get_ball_pos(self) -> torch.Tensor:
        if self.ball is None:
            return self.ball_pos
        return _as_torch(self.ball.get_pos(), self.device).reshape(self.num_envs, 3)

    def _get_foot_pos(self) -> torch.Tensor:
        if self.robot is None or not self.foot_link_idx:
            return torch.zeros(self.num_envs, 0, 3, device=self.device)
        return _as_torch(self.robot.get_links_pos(self.foot_link_idx), self.device).reshape(
            self.num_envs, len(self.foot_link_idx), 3
        )

    def _ball_foot_contact(self) -> torch.Tensor:
        feet = self._get_foot_pos()
        if feet.shape[1] == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        distances = torch.linalg.norm(feet - self.ball_pos[:, None, :], dim=-1)
        return torch.any(distances < float(self.reward_cfg["thresholds"]["contact_distance_m"]), dim=-1)

    def _control_torque(self, torque: torch.Tensor) -> None:
        if self.robot is not None:
            self.robot.control_dofs_force(torque, self.dof_idx)

    def _fallen(self, base_pos: torch.Tensor, base_rpy: torch.Tensor) -> torch.Tensor:
        safety = self.train_cfg["safety"]
        rp = torch.any(torch.abs(base_rpy[:, :2]) > float(safety["max_roll_pitch_rad"]), dim=-1)
        low = base_pos[:, 2] < float(safety["min_base_height"])
        high = base_pos[:, 2] > float(safety["max_base_height"])
        return rp | low | high

    def _is_kickable(self, ball_pos: torch.Tensor) -> torch.Tensor:
        task = self.train_cfg["task"]
        xr = task["kickable_x_range"]
        y_abs = float(task["kickable_y_abs_max"])
        return (ball_pos[:, 0] >= xr[0]) & (ball_pos[:, 0] <= xr[1]) & (torch.abs(ball_pos[:, 1]) <= y_abs)

    def _make_action_scale(self) -> torch.Tensor:
        control = self.train_cfg["control"]
        values = []
        for name in self.joint_names:
            if "shoulder" in name or "upper_arm" in name or "elbow" in name:
                values.append(float(control["arm_action_scale"]))
            elif "hip" in name or "thigh" in name or "calf" in name or "ankle" in name:
                values.append(float(control["leg_action_scale"]))
            else:
                values.append(float(control["action_scale_default"]))
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _make_max_torque(self) -> torch.Tensor:
        control = self.train_cfg["control"]
        values = []
        for name in self.joint_names:
            values.append(float(control["max_torque_arm"] if ("shoulder" in name or "upper_arm" in name or "elbow" in name) else control["max_torque_leg"]))
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _make_safety_limits(self) -> SafetyLimits:
        control = self.train_cfg["control"]
        safety = self.train_cfg["safety"]
        return SafetyLimits(
            max_delta_rad=float(control["max_delta_rad"]),
            max_delta_rate_rad_per_s=float(control["max_delta_rate_rad_per_s"]),
            max_joint_velocity_rad_per_s=float(control["max_joint_velocity_rad_per_s"]),
            max_roll_pitch_rad=float(safety["max_roll_pitch_rad"]),
            min_base_height=float(safety["min_base_height"]),
            max_base_height=float(safety["max_base_height"]),
            control_dt=self.control_dt,
            emergency_stop_file=safety.get("emergency_stop_file"),
        )

    def _dof_indices_by_joint_name(self) -> list[int]:
        by_name = {joint.name: joint for joint in self.robot.joints}
        return [int(by_name[name].dofs_idx_local[0]) for name in self.joint_names]

    def _base_link_idx(self) -> int:
        base_name = self.model_info.base_link_name or "base_link"
        for i, link in enumerate(self.robot.links):
            if link.name == base_name:
                return i
        return BASE_LINK_LOCAL_IDX_FALLBACK

    def _link_indices(self, names: list[str]) -> list[int]:
        by_name = {link.name: i for i, link in enumerate(self.robot.links)}
        return [by_name[name] for name in names if name in by_name]

    def _set_dofs_position(self, pos: torch.Tensor, env_ids: torch.Tensor) -> None:
        try:
            self.robot.set_dofs_position(pos, self.dof_idx, envs_idx=env_ids, zero_velocity=True)
        except TypeError:
            self.robot.set_dofs_position(pos[0], self.dof_idx, zero_velocity=True)

    def _set_entity_pos(self, entity, pos: torch.Tensor, env_ids: torch.Tensor) -> None:
        try:
            entity.set_pos(pos, envs_idx=env_ids)
        except TypeError:
            entity.set_pos(pos[0])

    def _curriculum_flag(self, key: str) -> bool:
        stage = int(self.domain_rand_cfg.get("curriculum", {}).get("stage", 0))
        stages = self.domain_rand_cfg.get("curriculum", {}).get("stages", [])
        if not stages:
            return False
        return bool(stages[min(stage, len(stages) - 1)].get(key, False))

    def _stage_value(self, key: str, default):
        stage = int(self.domain_rand_cfg.get("curriculum", {}).get("stage", 0))
        stages = self.domain_rand_cfg.get("curriculum", {}).get("stages", [])
        if not stages:
            return default
        return stages[min(stage, len(stages) - 1)].get(key, default)

    def _add_noise(self, value: torch.Tensor, key: str) -> torch.Tensor:
        if not self.domain_rand_cfg.get("enabled", False):
            return value
        std = float(self.domain_rand_cfg.get("noise", {}).get(key, 0.0) or 0.0)
        if std <= 0.0:
            return value
        return value + torch.randn_like(value) * std


def default_kick_env_paths(root: Path) -> KickEnvPaths:
    return KickEnvPaths(
        robot_cfg=root / "configs/pi_plus_genesis.yaml",
        train_cfg=root / "configs/pi_plus_kick_train.yaml",
        reward_cfg=root / "configs/pi_plus_kick_rewards.yaml",
        domain_rand_cfg=root / "configs/pi_plus_domain_randomization.yaml",
    )


def _resolve_torch_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def _rand_uniform(lo: float, hi: float, n: int, device: torch.device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(float(lo), float(hi))


def _as_torch(value, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(to_numpy(value), dtype=torch.float32, device=device)
