"""Safety filters for residual joint-angle kick policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SafetyLimits:
    max_delta_rad: float
    max_delta_rate_rad_per_s: float
    max_joint_velocity_rad_per_s: float
    max_roll_pitch_rad: float
    action_max_roll_pitch_rad: float
    min_base_height: float
    max_base_height: float
    action_min_base_height: float
    control_dt: float
    emergency_stop_file: str | None = None


class ResidualSafetyFilter:
    """Clamp residual actions before they reach low-level PD control."""

    def __init__(self, limits: SafetyLimits, action_scale: torch.Tensor):
        self.limits = limits
        self.action_scale = action_scale

    def filter_action(
        self,
        raw_action: torch.Tensor,
        prev_delta: torch.Tensor,
        joint_vel: torch.Tensor,
        base_rpy: torch.Tensor,
        base_height: torch.Tensor,
        kickable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.emergency_stop_requested():
            safe = torch.zeros_like(raw_action)
            return safe, safe

        delta = torch.clamp(raw_action, -1.0, 1.0) * self.action_scale
        delta = torch.clamp(delta, -self.limits.max_delta_rad, self.limits.max_delta_rad)

        max_step = self.limits.max_delta_rate_rad_per_s * self.limits.control_dt
        delta = torch.max(torch.min(delta, prev_delta + max_step), prev_delta - max_step)

        velocity_ok = torch.all(torch.abs(joint_vel) <= self.limits.max_joint_velocity_rad_per_s, dim=-1)
        attitude_ok = torch.all(torch.abs(base_rpy[:, :2]) <= self.limits.action_max_roll_pitch_rad, dim=-1)
        height_ok = (base_height >= self.limits.action_min_base_height) & (base_height <= self.limits.max_base_height)
        ok = velocity_ok & attitude_ok & height_ok & kickable
        safe = torch.where(ok[:, None], delta, torch.zeros_like(delta))
        return safe, delta

    def emergency_stop_requested(self) -> bool:
        if not self.limits.emergency_stop_file:
            return False
        return Path(self.limits.emergency_stop_file).exists()
