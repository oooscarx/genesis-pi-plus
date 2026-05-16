"""Reward terms for pi_plus ball kicking."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .math_utils import normalize_xy


@dataclass(frozen=True)
class KickRewardScales:
    ball_velocity_to_target: float
    ball_speed_match: float
    final_target_distance: float
    ball_contact: float
    foot_ball_proximity: float
    foot_ball_distance: float
    foot_ball_closer: float
    ball_escape: float
    base_upright: float
    base_height: float
    support_stability: float
    action_rate: float
    action_magnitude: float
    torque: float
    joint_limit: float
    fall: float
    not_kickable: float


def compute_kick_rewards(
    *,
    scales: KickRewardScales,
    ball_pos: torch.Tensor,
    prev_ball_pos: torch.Tensor,
    target_pos: torch.Tensor,
    desired_ball_speed: torch.Tensor,
    base_rpy: torch.Tensor,
    base_height: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    torque: torch.Tensor,
    contact: torch.Tensor,
    has_contacted_ball: torch.Tensor,
    foot_ball_distance: torch.Tensor,
    prev_foot_ball_distance: torch.Tensor,
    kickable: torch.Tensor,
    fallen: torch.Tensor,
    control_dt: float,
    base_height_target: float,
    base_height_sigma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ball_vel = (ball_pos[:, :2] - prev_ball_pos[:, :2]) / control_dt
    target_vec = target_pos[:, :2] - ball_pos[:, :2]
    target_dir = normalize_xy(target_vec)
    speed_to_target = torch.sum(ball_vel * target_dir, dim=-1)
    speed_mag = torch.linalg.norm(ball_vel, dim=-1)
    base_upright = torch.exp(-3.0 * torch.sum(torch.square(base_rpy[:, :2]), dim=-1))
    base_height_reward = torch.exp(-torch.square(base_height - base_height_target) / max(base_height_sigma**2, 1.0e-6))
    stable_gate = ((~fallen).float() * base_upright * base_height_reward).detach()
    pre_contact_gate = stable_gate * (~has_contacted_ball).float()
    post_contact_gate = stable_gate * has_contacted_ball.float()
    foot_ball_delta = prev_foot_ball_distance - foot_ball_distance

    terms = {
        "ball_velocity_to_target": post_contact_gate * torch.clamp(speed_to_target, min=0.0),
        "ball_speed_match": post_contact_gate * torch.exp(-torch.square(speed_mag - desired_ball_speed)),
        "final_target_distance": post_contact_gate * torch.exp(-torch.linalg.norm(target_vec, dim=-1)),
        "ball_contact": stable_gate * contact.float(),
        "foot_ball_proximity": pre_contact_gate * torch.exp(-torch.square(foot_ball_distance / 0.12)),
        "foot_ball_distance": pre_contact_gate * foot_ball_distance,
        "foot_ball_closer": pre_contact_gate * torch.clamp(foot_ball_delta / 0.05, min=-1.0, max=1.0),
        "ball_escape": ((~has_contacted_ball) & (foot_ball_distance > 0.45)).float(),
        "base_upright": base_upright,
        "base_height": base_height_reward,
        "support_stability": torch.exp(-torch.linalg.norm(base_rpy[:, :2], dim=-1)),
        "action_rate": torch.mean(torch.square(action - prev_action), dim=-1),
        "action_magnitude": torch.mean(torch.square(action), dim=-1),
        "torque": torch.mean(torch.square(torque), dim=-1),
        "joint_limit": torch.zeros_like(base_height),
        "fall": fallen.float(),
        "not_kickable": (~kickable).float(),
    }

    reward = torch.zeros_like(base_height)
    for name, value in terms.items():
        reward = reward + getattr(scales, name) * value
    return reward, terms
