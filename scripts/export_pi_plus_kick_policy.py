#!/usr/bin/env python3
"""Export a trained pi_plus kick actor to TorchScript and optionally ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.kick_env import PiPlusKickEnv, default_kick_env_paths
from genesis_pi_plus.policy_io import export_actor


def main() -> None:
    args = parse_args()
    env = PiPlusKickEnv(default_kick_env_paths(ROOT), num_envs=1, build_scene=False, device="cpu")
    export_actor(args.checkpoint, args.output, env.num_obs, onnx_path=args.onnx)
    print(f"exported TorchScript actor: {args.output}")
    if args.onnx:
        print(f"exported ONNX actor: {args.onnx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pi_plus kick policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="exports/pi_plus_kick_policy.pt")
    parser.add_argument("--onnx", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
