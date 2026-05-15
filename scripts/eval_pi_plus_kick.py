#!/usr/bin/env python3
"""Evaluate pi_plus residual kick policy success/fall metrics."""

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
        num_envs=args.num_envs,
        device=args.device,
        backend=args.backend,
        headless=True,
        build_scene=True,
    )
    policy = load_actor_from_checkpoint(args.checkpoint, env.device)
    obs, _ = env.reset()
    reward_sum = torch.zeros(env.num_envs, device=env.device)
    done_count = torch.zeros(env.num_envs, device=env.device)
    for _ in range(int(args.duration / env.control_dt)):
        with torch.no_grad():
            action = policy(obs)
        obs, reward, done, extras = env.step(action)
        reward_sum += reward
        done_count += done.float()
    success = extras["log"].get("episode/success_proxy")
    fall_rate = extras["log"].get("episode/fall_rate")
    print(f"mean_reward={reward_sum.mean().item():.4f}")
    print(f"done_count_mean={done_count.mean().item():.4f}")
    print(f"success_proxy={float(success):.4f}")
    print(f"fall_rate={float(fall_rate):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pi_plus kick policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    main()
