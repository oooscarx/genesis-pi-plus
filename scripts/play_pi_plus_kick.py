#!/usr/bin/env python3
"""Visualize a trained pi_plus residual kick policy in Genesis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.kick_env import PiPlusKickEnv, default_kick_env_paths
from genesis_pi_plus.policy_io import load_actor_from_checkpoint


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
    policy = load_actor_from_checkpoint(args.checkpoint, env.device) if args.checkpoint else None
    obs, _ = env.reset()
    steps = int(args.duration / env.control_dt)
    for _ in range(steps):
        if policy is None:
            action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        else:
            with torch.no_grad():
                action = policy(obs)
        obs, reward, done, extras = env.step(action)
        if bool(done[0]):
            obs, _ = env.reset()
    print("play finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play pi_plus kick policy.")
    parser.add_argument("--checkpoint", default=None, help="rsl-rl checkpoint or exported TorchScript actor.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
