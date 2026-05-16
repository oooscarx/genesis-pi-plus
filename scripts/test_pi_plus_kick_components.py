#!/usr/bin/env python3
"""Fast non-Genesis checks for kick env observations, safety, and rewards."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.kick_env import PiPlusKickEnv, default_kick_env_paths
from genesis_pi_plus.kick_rewards import KickRewardScales, compute_kick_rewards


def main() -> None:
    test_observation_shape()
    test_safety_action_clamp()
    test_reward_direction_term()
    test_reward_contact_stage_gating()
    print("kick component tests passed")


def test_observation_shape() -> None:
    env = PiPlusKickEnv(default_kick_env_paths(ROOT), num_envs=3, device="cpu", build_scene=False)
    obs, extras = env.reset()
    assert obs.shape == (3, env.num_obs)
    assert env.num_obs == 75
    assert "critic" in extras["observations"]


def test_safety_action_clamp() -> None:
    env = PiPlusKickEnv(default_kick_env_paths(ROOT), num_envs=2, device="cpu", build_scene=False)
    env._foot_ball_contact = lambda: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    raw = torch.full((2, env.num_actions), 10.0)
    obs, _ = env.reset()
    _, _, _, _ = env.step(raw)
    assert torch.all(torch.abs(env.actions) <= env.train_cfg["control"]["max_delta_rad"] + 1.0e-6)


def test_reward_direction_term() -> None:
    scales = KickRewardScales(
        ball_velocity_to_target=1.0,
        ball_speed_match=0.0,
        final_target_distance=0.0,
        ball_contact=0.0,
        foot_ball_proximity=0.0,
        foot_ball_distance=0.0,
        foot_ball_closer=0.0,
        ball_escape=0.0,
        base_upright=0.0,
        base_height=0.0,
        support_stability=0.0,
        action_rate=0.0,
        action_magnitude=0.0,
        torque=0.0,
        joint_limit=0.0,
        fall=0.0,
        not_kickable=0.0,
    )
    reward_forward, _ = compute_kick_rewards(
        scales=scales,
        ball_pos=torch.tensor([[0.1, 0.0, 0.05]]),
        prev_ball_pos=torch.tensor([[0.0, 0.0, 0.05]]),
        target_pos=torch.tensor([[1.0, 0.0]]),
        desired_ball_speed=torch.tensor([1.0]),
        base_rpy=torch.zeros(1, 3),
        base_height=torch.tensor([0.38]),
        action=torch.zeros(1, 20),
        prev_action=torch.zeros(1, 20),
        torque=torch.zeros(1, 20),
        contact=torch.tensor([False]),
        has_contacted_ball=torch.tensor([True]),
        foot_ball_distance=torch.tensor([1.0]),
        prev_foot_ball_distance=torch.tensor([1.0]),
        kickable=torch.tensor([True]),
        fallen=torch.tensor([False]),
        control_dt=0.1,
        base_height_target=0.38,
        base_height_sigma=0.1,
    )
    reward_backward, _ = compute_kick_rewards(
        scales=scales,
        ball_pos=torch.tensor([[-0.1, 0.0, 0.05]]),
        prev_ball_pos=torch.tensor([[0.0, 0.0, 0.05]]),
        target_pos=torch.tensor([[1.0, 0.0]]),
        desired_ball_speed=torch.tensor([1.0]),
        base_rpy=torch.zeros(1, 3),
        base_height=torch.tensor([0.38]),
        action=torch.zeros(1, 20),
        prev_action=torch.zeros(1, 20),
        torque=torch.zeros(1, 20),
        contact=torch.tensor([False]),
        has_contacted_ball=torch.tensor([True]),
        foot_ball_distance=torch.tensor([1.0]),
        prev_foot_ball_distance=torch.tensor([1.0]),
        kickable=torch.tensor([True]),
        fallen=torch.tensor([False]),
        control_dt=0.1,
        base_height_target=0.38,
        base_height_sigma=0.1,
    )
    assert reward_forward.item() > reward_backward.item()


def test_reward_contact_stage_gating() -> None:
    scales = KickRewardScales(
        ball_velocity_to_target=1.0,
        ball_speed_match=0.0,
        final_target_distance=0.0,
        ball_contact=0.0,
        foot_ball_proximity=0.0,
        foot_ball_distance=-1.0,
        foot_ball_closer=0.0,
        ball_escape=-1.0,
        base_upright=0.0,
        base_height=0.0,
        support_stability=0.0,
        action_rate=0.0,
        action_magnitude=0.0,
        torque=0.0,
        joint_limit=0.0,
        fall=0.0,
        not_kickable=0.0,
    )
    common = dict(
        scales=scales,
        ball_pos=torch.tensor([[1.0, 0.0, 0.05]]),
        prev_ball_pos=torch.tensor([[0.9, 0.0, 0.05]]),
        target_pos=torch.tensor([[2.0, 0.0]]),
        desired_ball_speed=torch.tensor([1.0]),
        base_rpy=torch.zeros(1, 3),
        base_height=torch.tensor([0.38]),
        action=torch.zeros(1, 20),
        prev_action=torch.zeros(1, 20),
        torque=torch.zeros(1, 20),
        contact=torch.tensor([False]),
        foot_ball_distance=torch.tensor([1.0]),
        prev_foot_ball_distance=torch.tensor([0.9]),
        kickable=torch.tensor([True]),
        fallen=torch.tensor([False]),
        control_dt=0.1,
        base_height_target=0.38,
        base_height_sigma=0.1,
    )
    pre_reward, pre_terms = compute_kick_rewards(has_contacted_ball=torch.tensor([False]), **common)
    post_reward, post_terms = compute_kick_rewards(has_contacted_ball=torch.tensor([True]), **common)
    assert pre_terms["ball_velocity_to_target"].item() == 0.0
    assert pre_terms["foot_ball_distance"].item() > 0.0
    assert post_terms["ball_velocity_to_target"].item() > 0.0
    assert post_terms["foot_ball_distance"].item() == 0.0
    assert post_reward.item() > pre_reward.item()


if __name__ == "__main__":
    main()
