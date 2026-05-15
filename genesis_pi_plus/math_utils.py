"""Small math helpers used by policy rollout and kick training."""

from __future__ import annotations

import math

import numpy as np
import torch


def quat_rotate_inverse_wxyz_torch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by inverse quaternions in wxyz format."""
    q_vec = q[..., 1:4]
    w = q[..., 0:1]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v - 2.0 * (w * uv + uuv)


def quat_rotate_wxyz_torch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in wxyz format."""
    q_vec = q[..., 1:4]
    w = q[..., 0:1]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (w * uv + uuv)


def quat_wxyz_to_rpy_np(q: np.ndarray) -> tuple[float, float, float]:
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


def quat_wxyz_to_rpy_torch(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def normalize_xy(v: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
