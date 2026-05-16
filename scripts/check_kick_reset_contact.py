#!/usr/bin/env python3
"""Diagnose whether pi_plus kick contact availability survives repeated resets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.kick_env import PiPlusKickEnv, default_kick_env_paths


@torch.no_grad()
def main() -> None:
    args = parse_args()
    env = PiPlusKickEnv(
        default_kick_env_paths(ROOT),
        num_envs=args.num_envs,
        device=args.device,
        backend=args.backend,
        headless=True,
        build_scene=True,
    )
    steps = args.steps or env.max_episode_length
    print(
        f"reset/contact diagnostic: num_envs={env.num_envs} episodes={args.episodes} "
        f"steps={steps} mode={args.mode} device={env.device}"
    )

    for episode in range(args.episodes):
        env.reset()
        initial_distance = env._foot_ball_distance().mean().item()
        initial_has_contacted = env.has_contacted_ball.float().mean().item()
        initial_contact = env._foot_ball_contact().float().mean().item()

        contact_peak = 0.0
        has_contacted_peak = 0.0
        fall_peak = 0.0
        escape_peak = 0.0
        kick_window_mean = 0.0
        action_mag_mean = 0.0
        min_distance = float("inf")
        final_distance = initial_distance

        for _ in range(steps):
            actions = make_actions(env, args.mode)
            _obs, _rewards, _done, extras = env.step(actions)
            log = extras["log"]
            contact_peak = max(contact_peak, float(log["episode/contact_rate"]))
            has_contacted_peak = max(has_contacted_peak, float(log["episode/has_contacted_ball_rate"]))
            fall_peak = max(fall_peak, float(log["episode/fall_rate"]))
            escape_peak = max(escape_peak, float(log["episode/ball_escape_rate"]))
            kick_window_mean += float(log["metric/kick_window_active"]) / steps
            action_mag_mean += float(torch.mean(torch.square(env.actions))) / steps
            distance = float(log["metric/foot_ball_distance_m"])
            min_distance = min(min_distance, distance)
            final_distance = distance

        print(
            f"episode={episode:02d} "
            f"initial_distance={initial_distance:.4f} "
            f"initial_contact={initial_contact:.4f} "
            f"initial_has_contacted={initial_has_contacted:.4f} "
            f"min_distance={min_distance:.4f} "
            f"final_distance={final_distance:.4f} "
            f"contact_peak={contact_peak:.4f} "
            f"has_contacted_peak={has_contacted_peak:.4f} "
            f"fall_peak={fall_peak:.4f} "
            f"escape_peak={escape_peak:.4f} "
            f"kick_window_mean={kick_window_mean:.4f} "
            f"action_mag_mean={action_mag_mean:.6f}"
        )


def make_actions(env: PiPlusKickEnv, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(env.num_envs, env.num_actions, device=env.device)
    if mode == "random":
        return torch.randn(env.num_envs, env.num_actions, device=env.device)
    if mode == "right_leg_sweep":
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for idx, name in enumerate(env.joint_names):
            if name.startswith("r_") and any(part in name for part in ("hip", "thigh", "calf", "ankle")):
                actions[:, idx] = 1.0
        return actions
    raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--mode", choices=["random", "zero", "right_leg_sweep"], default="random")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    main()
