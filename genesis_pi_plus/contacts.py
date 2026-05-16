"""Contact query helpers isolated from policy code."""

from __future__ import annotations

from typing import Any

import torch


def get_foot_ball_contacts(robot: Any, ball: Any, foot_link_global_idx: list[int], device: torch.device) -> torch.Tensor:
    """Return per-env true foot-ball contact flags from Genesis contact pairs.

    Genesis reports contacts from the most recent ``scene.step()``. Link indices
    in the returned contact dict are global rigid-solver link indices, not
    per-entity local indices.
    """
    if not foot_link_global_idx:
        return torch.zeros(1, dtype=torch.bool, device=device)
    contact_info = robot.get_contacts(with_entity=ball)
    link_a = _as_2d_tensor(contact_info.get("link_a"), device)
    link_b = _as_2d_tensor(contact_info.get("link_b"), device)
    if link_a.numel() == 0 or link_b.numel() == 0:
        return torch.zeros(link_a.shape[0] if link_a.ndim else 1, dtype=torch.bool, device=device)

    valid_mask = contact_info.get("valid_mask")
    if valid_mask is None:
        valid = torch.ones_like(link_a, dtype=torch.bool, device=device)
    else:
        valid = _as_2d_tensor(valid_mask, device).bool()

    foot_ids = torch.tensor(foot_link_global_idx, dtype=link_a.dtype, device=device)
    foot_contact = _isin_last_dim(link_a, foot_ids) | _isin_last_dim(link_b, foot_ids)
    return torch.any(valid & foot_contact, dim=-1)


def get_foot_contacts(scene: Any, robot: Any, foot_link_names: list[str]):
    raise NotImplementedError("Use get_foot_ball_contacts(...) with explicit Genesis global foot link indices.")


def get_ball_contacts(scene: Any, ball: Any, robot: Any | None = None):
    raise NotImplementedError("TODO: expose generic ball contact summaries once needed by rewards/eval.")


def _as_2d_tensor(value: Any, device: torch.device) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _isin_last_dim(values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    if candidates.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return torch.any(values[..., None] == candidates, dim=-1)
