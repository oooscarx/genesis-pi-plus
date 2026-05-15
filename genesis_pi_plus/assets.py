"""Path helpers for local assets referenced from this repository."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the genesis_pi_plus repository root."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | Path | None) -> Path | None:
    """Resolve a config path relative to the repository root."""
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def ensure_exists(path: str | Path | None) -> Path:
    """Resolve a path and raise a clear error if it does not exist."""
    resolved = resolve_path(path)
    if resolved is None:
        raise ValueError("Path is null. Fill configs/pi_plus_genesis.yaml first.")
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved
