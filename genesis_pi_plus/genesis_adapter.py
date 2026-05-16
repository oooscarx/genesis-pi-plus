"""Thin Genesis API boundary for pi_plus.

All direct Genesis calls live here so uncertain APIs are isolated from the
controller, observation, and task code.
"""

from __future__ import annotations

import os
import platform
from typing import Any

import numpy as np

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
    backend_name = (backend or os.environ.get("GENESIS_BACKEND") or "cuda").lower()
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
        surface = _make_ground_surface(gs, scene_cfg)
        return scene.add_entity(ground, material=material, surface=surface, name="ground")
    except Exception as exc:
        raise RuntimeError("TODO: Verify Genesis ground plane API.") from exc


def _make_ground_surface(gs: Any, scene_cfg: dict[str, Any]):
    color = tuple(scene_cfg.get("ground_color", [0.10, 0.45, 0.16, 1.0]))
    if scene_cfg.get("field_style") != "soccer":
        return gs.surfaces.Rough(color=color)

    try:
        texture = _make_soccer_field_texture(scene_cfg)
        return gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_array=texture, encoding="srgb"))
    except Exception as exc:
        raise RuntimeError("TODO: Verify Genesis ImageTexture support for the soccer field surface.") from exc


def _make_soccer_field_texture(scene_cfg: dict[str, Any]) -> np.ndarray:
    """Generate a simple soccer pitch texture as RGBA uint8."""
    width, height = [int(v) for v in scene_cfg.get("field_texture_resolution", [1024, 768])]
    plane_x, plane_y = [float(v) for v in scene_cfg.get("ground_plane_size", [12.0, 8.0])]
    base = np.array(scene_cfg.get("ground_color", [0.10, 0.45, 0.16, 1.0]), dtype=np.float32)
    alt = np.clip(base * np.array([0.78, 1.12, 0.82, 1.0], dtype=np.float32), 0.0, 1.0)
    line = np.array(scene_cfg.get("field_line_color", [0.93, 0.95, 0.90, 1.0]), dtype=np.float32)
    img = np.zeros((height, width, 4), dtype=np.float32)

    stripes = max(1, int(scene_cfg.get("field_stripe_count", 10)))
    stripe_width = max(1, width // stripes)
    for x in range(width):
        img[:, x, :] = base if (x // stripe_width) % 2 == 0 else alt

    def px_x(x_m: float) -> int:
        return int(round((x_m / plane_x + 0.5) * (width - 1)))

    def px_y(y_m: float) -> int:
        return int(round((0.5 - y_m / plane_y) * (height - 1)))

    line_px = max(2, int(round(float(scene_cfg.get("field_line_width_m", 0.045)) / plane_x * width)))

    def draw_rect(x0: float, y0: float, x1: float, y1: float) -> None:
        x0p, x1p = sorted((px_x(x0), px_x(x1)))
        y0p, y1p = sorted((px_y(y0), px_y(y1)))
        img[y0p : y0p + line_px, x0p:x1p, :] = line
        img[y1p - line_px : y1p, x0p:x1p, :] = line
        img[y0p:y1p, x0p : x0p + line_px, :] = line
        img[y0p:y1p, x1p - line_px : x1p, :] = line

    def draw_vline(x: float, y0: float, y1: float) -> None:
        xp = px_x(x)
        y0p, y1p = sorted((px_y(y0), px_y(y1)))
        img[y0p:y1p, max(0, xp - line_px // 2) : min(width, xp + line_px // 2 + 1), :] = line

    def draw_circle(cx: float, cy: float, radius: float) -> None:
        yy, xx = np.ogrid[:height, :width]
        cxp, cyp = px_x(cx), px_y(cy)
        rx = radius / plane_x * width
        ry = radius / plane_y * height
        dist = ((xx - cxp) / rx) ** 2 + ((yy - cyp) / ry) ** 2
        thickness = max(0.006, line_px / max(width, height) * 2.0)
        mask = np.abs(dist - 1.0) <= thickness
        img[mask] = line

    margin_x = plane_x * 0.055
    margin_y = plane_y * 0.075
    half_x = plane_x / 2 - margin_x
    half_y = plane_y / 2 - margin_y
    draw_rect(-half_x, -half_y, half_x, half_y)
    draw_vline(0.0, -half_y, half_y)
    draw_circle(0.0, 0.0, min(plane_x, plane_y) * 0.11)
    draw_rect(-half_x, -half_y * 0.42, -half_x + plane_x * 0.16, half_y * 0.42)
    draw_rect(half_x - plane_x * 0.16, -half_y * 0.42, half_x, half_y * 0.42)
    draw_rect(-half_x, -half_y * 0.22, -half_x + plane_x * 0.055, half_y * 0.22)
    draw_rect(half_x - plane_x * 0.055, -half_y * 0.22, half_x, half_y * 0.22)
    img[px_y(0.0) - line_px : px_y(0.0) + line_px + 1, px_x(0.0) - line_px : px_x(0.0) + line_px + 1, :] = line
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


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
            surface=gs.surfaces.Rough(color=tuple(ball_cfg.get("color", [0.96, 0.94, 0.86, 1.0]))),
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
