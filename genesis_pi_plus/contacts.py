"""Contact query placeholders isolated from policy code."""

from __future__ import annotations

from typing import Any


def get_foot_contacts(scene: Any, robot: Any, foot_link_names: list[str]):
    raise NotImplementedError("TODO: verify Genesis contact API and foot link identifiers.")


def get_ball_contacts(scene: Any, ball: Any, robot: Any | None = None):
    raise NotImplementedError("TODO: verify Genesis contact API for ball-ground and ball-robot contacts.")
