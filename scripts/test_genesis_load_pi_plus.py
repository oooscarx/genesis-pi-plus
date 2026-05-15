#!/usr/bin/env python3
"""Smoke test: load pi_plus into Genesis without opening a viewer."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.config import load_config
from genesis_pi_plus.genesis_adapter import add_ground, create_scene, init_genesis, load_pi_plus


def main() -> None:
    cfg = load_config(ROOT / "configs/pi_plus_genesis.yaml")
    cfg.setdefault("sim", {})["headless"] = True
    if cfg.get("robot", {}).get("asset_file") is None:
        raise SystemExit(
            "robot.asset_file is null. Run `uv run python scripts/inspect_amp_tk.py` "
            "and fill configs/pi_plus_genesis.yaml with a ../AMP_TK/... path."
        )

    init_genesis(headless=True, backend=cfg.get("sim", {}).get("backend"))
    scene = create_scene(cfg)
    add_ground(scene, cfg)
    robot = load_pi_plus(scene, cfg)

    print("Loaded pi_plus into Genesis.")
    print_entity_info("robot", robot)


def print_entity_info(label: str, entity) -> None:
    print(f"{label}: {entity!r}")
    for attr in ("n_links", "n_joints", "n_dofs", "links", "joints", "dofs"):
        if hasattr(entity, attr):
            value = getattr(entity, attr)
            print(f"{label}.{attr}: {value}")
    print("TODO: verify exact Genesis link/joint introspection API on the target host.")


if __name__ == "__main__":
    main()
