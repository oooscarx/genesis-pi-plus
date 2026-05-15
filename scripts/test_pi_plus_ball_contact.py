#!/usr/bin/env python3
"""Smoke test scaffold: load pi_plus, ground, and a ball in Genesis."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.config import load_config
from genesis_pi_plus.contacts import get_ball_contacts
from genesis_pi_plus.genesis_adapter import add_ball, add_ground, create_scene, init_genesis, load_pi_plus, step_scene


def main() -> None:
    cfg = load_config(ROOT / "configs/pi_plus_genesis.yaml")
    cfg.setdefault("sim", {})["headless"] = True

    init_genesis(headless=True, backend=cfg.get("sim", {}).get("backend"))
    scene = create_scene(cfg)
    add_ground(scene, cfg)
    robot = load_pi_plus(scene, cfg)
    ball = add_ball(scene, cfg)

    step_scene(scene, 240)
    print(f"robot: {robot!r}")
    print(f"ball: {ball!r}")
    print("TODO: read ball position/velocity from Genesis entity API.")
    try:
        print(get_ball_contacts(scene, ball, robot))
    except NotImplementedError as exc:
        print(exc)


if __name__ == "__main__":
    main()
