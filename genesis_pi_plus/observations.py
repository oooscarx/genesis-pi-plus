"""Observation placeholders for future pi_plus ball-kick RL."""

from __future__ import annotations

from typing import Any


def get_base_rpy(robot: Any):
    raise NotImplementedError("TODO: map Genesis pi_plus base orientation API to roll/pitch/yaw.")


def get_base_angular_velocity(robot: Any):
    raise NotImplementedError("TODO: map Genesis pi_plus base angular velocity API.")


def get_joint_pos_vel(robot: Any, joint_names: list[str]):
    raise NotImplementedError("TODO: read Genesis joint position/velocity in configured joint order.")


def get_ball_position_in_robot_frame(robot: Any, ball: Any):
    raise NotImplementedError("TODO: compute ball position in pi_plus base frame after Genesis API is verified.")


def get_foot_contact_observation(robot: Any, scene: Any, foot_link_names: list[str]):
    raise NotImplementedError("TODO: expose foot contact state from Genesis contact API.")
