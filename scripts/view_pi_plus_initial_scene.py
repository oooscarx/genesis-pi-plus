#!/usr/bin/env python3
"""Open a Genesis viewer for the pi_plus initial kick scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.kick_env import PiPlusKickEnv, default_kick_env_paths


def main() -> None:
    args = parse_args()
    env = PiPlusKickEnv(
        default_kick_env_paths(ROOT),
        num_envs=1,
        device=args.device,
        backend=args.backend,
        headless=not args.viewer,
        build_scene=True,
    )
    obs, _ = env.reset()
    del obs

    contact_threshold = float(env.reward_cfg["thresholds"]["contact_distance_m"])
    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    steps = max(1, int(args.duration / env.control_dt))

    print_initial_state(env, contact_threshold)
    print(f"Viewer: {'on' if args.viewer else 'off'}, duration={args.duration:.2f}s")

    for step in range(steps):
        _obs, _reward, done, extras = env.step(action)
        if step % max(1, args.log_every) == 0:
            ball_pos = env.ball_pos[0].detach().cpu().tolist()
            distance = float(extras["log"]["metric/foot_ball_distance_m"])
            contact = float(extras["log"]["episode/contact_rate"])
            has_contacted = float(extras["log"]["episode/has_contacted_ball_rate"])
            print(
                f"t={step * env.control_dt:5.2f}s "
                f"ball=({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f}) "
                f"foot_ball_distance={distance:.4f} "
                f"contact={contact:.0f} has_contacted={has_contacted:.0f}"
            )
        if bool(done[0]):
            print("Episode reset triggered while viewing.")
            env.reset()


def print_initial_state(env: PiPlusKickEnv, contact_threshold: float) -> None:
    ball_pos = env.ball_pos[0].detach().cpu().tolist()
    distance = float(env._foot_ball_distance()[0])
    print("Initial pi_plus kick scene")
    print(f"ball_pos_robot=({ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f})")
    print(f"foot_ball_distance={distance:.4f} m")
    print(f"distance_fallback_contact_threshold={contact_threshold:.4f} m")
    print(f"distance_fallback_initial_contact={distance < contact_threshold}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View the initial pi_plus kick scene in Genesis.")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--backend", default=None, help="Genesis backend override: cpu, metal, cuda.")
    parser.add_argument("--device", default="cpu", help="Torch device for env tensors: cpu, mps, cuda.")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    main()
