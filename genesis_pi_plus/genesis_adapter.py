"""Thin Genesis API boundary for pi_plus.

All direct Genesis calls live here so uncertain APIs are isolated from the
controller, observation, and task code.
"""

from __future__ import annotations

import os
import platform
from typing import Any

from .assets import ensure_exists


def _configure_headless_import(headless: bool = True) -> None:
    """Set renderer env vars before Genesis imports pyglet/pyrender."""
    if not headless:
        return
    if platform.system() == "Linux":
        os.environ.setdefault("GENESIS_HEADLESS", "1")
        os.environ.setdefault("PYGLET_HEADLESS", "true")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _import_genesis(headless: bool = True):
    _configure_headless_import(headless)
    try:
        import genesis as gs  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import Genesis. Run `uv sync` first; on headless Linux also "
            "install libegl1/libgl1/libxrender1 and related OpenGL/X11 runtime libs."
        ) from exc
    return gs


def init_genesis(headless: bool = True, backend: str | None = None):
    """Initialize Genesis with a headless-safe default."""
    gs = _import_genesis(headless=headless)
    backend_name = (os.environ.get("GENESIS_BACKEND") or backend or "cuda").lower()
    backend_obj = _select_backend(gs, backend_name)
    try:
        gs.init(backend=backend_obj)
    except TypeError:
        gs.init()
    print(f"Genesis initialized with backend={backend_obj}.")
    return gs


def _select_backend(gs: Any, backend_name: str):
    """Resolve a Genesis backend name, with CUDA fallback for local Mac tests."""
    if backend_name in {"cuda", "gpu"}:
        try:
            import torch

            if not torch.cuda.is_available():
                print("Requested Genesis CUDA backend, but torch.cuda is unavailable; falling back to CPU.")
                backend_name = "cpu"
        except ImportError:
            print("Torch is unavailable while checking CUDA; falling back to CPU.")
            backend_name = "cpu"

    if not hasattr(gs, backend_name):
        available = [name for name in ("cpu", "cuda", "gpu", "metal") if hasattr(gs, name)]
        raise ValueError(f"Unsupported Genesis backend '{backend_name}'. Available: {available}")
    return getattr(gs, backend_name)


def create_scene(cfg: dict[str, Any]):
    """Create a Genesis scene from the YAML config."""
    sim_cfg = cfg.get("sim", {})
    gs = _import_genesis(headless=bool(sim_cfg.get("headless", True)))
    scene_cfg = cfg.get("scene", {})
    headless = bool(sim_cfg.get("headless", True))
    sim_dt = sim_cfg.get("sim_dt")

    try:
        sim_options = gs.options.SimOptions(dt=sim_dt) if sim_dt is not None else None
        scene_kwargs: dict[str, Any] = {"show_viewer": not headless}
        if sim_options is not None:
            scene_kwargs["sim_options"] = sim_options
        if not headless:
            scene_kwargs["viewer_options"] = gs.options.ViewerOptions(
                camera_pos=tuple(scene_cfg.get("viewer_camera_pos", [2.2, -2.2, 1.4])),
                camera_lookat=tuple(scene_cfg.get("viewer_camera_lookat", [0.15, 0.0, 0.35])),
                camera_fov=float(scene_cfg.get("viewer_camera_fov", 45)),
                max_FPS=60,
            )
        return gs.Scene(**scene_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "TODO: Genesis Scene construction API needs verification for this "
            "installed genesis-world version."
        ) from exc


def load_pi_plus(scene, cfg: dict[str, Any]):
    """Load the pi_plus MJCF/URDF asset into a Genesis scene."""
    gs = _import_genesis(headless=bool(cfg.get("sim", {}).get("headless", True)))
    asset_file = cfg.get("robot", {}).get("asset_file")
    if asset_file is None:
        raise ValueError(
            "robot.asset_file is null. Run `uv run python scripts/inspect_amp_tk.py`, "
            "then fill configs/pi_plus_genesis.yaml with assets/pi_plus/pi_plus.xml or another asset path."
        )
    asset_path = ensure_exists(asset_file)
    suffix = asset_path.suffix.lower()

    try:
        if suffix == ".xml":
            morph = gs.morphs.MJCF(file=str(asset_path))
        elif suffix == ".urdf":
            morph = gs.morphs.URDF(file=str(asset_path))
        else:
            raise ValueError(f"Unsupported pi_plus asset type for Genesis load: {asset_path}")
        return scene.add_entity(morph)
    except Exception as exc:
        raise RuntimeError(
            "TODO: Verify Genesis robot loading API and MJCF mesh path handling. "
            f"Attempted to load {asset_path}."
        ) from exc


def add_ground(scene, cfg: dict[str, Any]):
    """Add a plane if enabled by config."""
    scene_cfg = cfg.get("scene", {})
    if not scene_cfg.get("ground", True):
        return None
    gs = _import_genesis(headless=bool(cfg.get("sim", {}).get("headless", True)))
    try:
        ground = gs.morphs.Plane(
            plane_size=tuple(scene_cfg.get("ground_plane_size", [20.0, 20.0])),
            tile_size=tuple(scene_cfg.get("ground_tile_size", [0.25, 0.25])),
        )
        material = gs.materials.Rigid(friction=scene_cfg.get("ground_friction", 1.0))
        surface = gs.surfaces.Rough(color=tuple(scene_cfg.get("ground_color", [0.58, 0.62, 0.60, 1.0])))
        return scene.add_entity(ground, material=material, surface=surface, name="ground")
    except Exception as exc:
        raise RuntimeError("TODO: Verify Genesis ground plane API.") from exc


def add_ball(scene, cfg: dict[str, Any]):
    """Add a lightweight soccer-size ball."""
    gs = _import_genesis(headless=bool(cfg.get("sim", {}).get("headless", True)))
    ball_cfg = cfg.get("ball", {})
    radius = float(ball_cfg.get("radius", 0.05))
    mass = float(ball_cfg.get("mass", 0.043))
    pos = ball_cfg.get("initial_pos", [0.18, -0.05, 0.05])
    try:
        volume = 4.0 / 3.0 * 3.141592653589793 * radius**3
        material_kwargs = {"rho": mass / volume}
        if ball_cfg.get("friction") is not None:
            material_kwargs["friction"] = ball_cfg["friction"]
        return scene.add_entity(
            gs.morphs.Sphere(radius=radius, pos=tuple(pos)),
            material=gs.materials.Rigid(**material_kwargs),
            surface=gs.surfaces.Rough(color=(0.92, 0.38, 0.12, 1.0)),
            name="ball",
        )
    except Exception as exc:
        raise RuntimeError(
            "TODO: Verify Genesis sphere API and mass/friction material settings."
        ) from exc


def step_scene(scene, n_steps: int):
    """Step a Genesis scene n times."""
    for _ in range(int(n_steps)):
        try:
            scene.step()
        except Exception as exc:
            raise RuntimeError("TODO: Verify Genesis scene stepping API.") from exc
