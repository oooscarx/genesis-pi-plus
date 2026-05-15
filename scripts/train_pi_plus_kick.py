#!/usr/bin/env python3
"""Train the pi_plus residual kick policy with Genesis + rsl-rl."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.config import load_config
from genesis_pi_plus.kick_env import PiPlusKickEnv, default_kick_env_paths


def main() -> None:
    args = parse_args()
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise SystemExit("rsl-rl-lib is not installed. Run `uv sync` after pulling the updated pyproject.toml.") from exc

    paths = default_kick_env_paths(ROOT)
    train_cfg = load_config(paths.train_cfg)
    if args.num_envs is not None:
        train_cfg["num_envs"] = args.num_envs

    env = PiPlusKickEnv(
        paths,
        num_envs=train_cfg["num_envs"],
        device=args.device or train_cfg.get("device", "cuda"),
        backend=args.backend,
        headless=True,
        build_scene=True,
    )
    runner_cfg = train_cfg["rsl_rl"]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=str(log_dir), device=str(env.device))
    runner.learn(num_learning_iterations=args.iterations or runner_cfg["max_iterations"], init_at_random_ep_len=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pi_plus residual kick policy.")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--backend", default=None, help="Genesis backend: cuda, gpu, cpu, metal.")
    parser.add_argument("--device", default=None, help="Torch device for rsl-rl: cuda or cpu.")
    parser.add_argument("--log-dir", default="runs/pi_plus_kick")
    return parser.parse_args()


if __name__ == "__main__":
    main()
