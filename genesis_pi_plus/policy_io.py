"""Policy checkpoint loading and export helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def build_mlp_from_actor_state(actor_state: dict[str, torch.Tensor]) -> nn.Sequential:
    """Build an actor MLP from rsl-rl-style actor.N.weight keys."""
    layer_ids = sorted(
        {
            int(key.split(".")[0])
            for key in actor_state
            if key.endswith(".weight") and key.split(".")[0].isdigit()
        }
    )
    layers: list[nn.Module] = []
    for i, layer_id in enumerate(layer_ids):
        weight = actor_state[f"{layer_id}.weight"]
        bias = actor_state[f"{layer_id}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if i != len(layer_ids) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def load_actor_from_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Load the actor network from an rsl-rl checkpoint or TorchScript file."""
    checkpoint_path = Path(path)
    try:
        policy = torch.jit.load(str(checkpoint_path), map_location=device)
        return policy.eval()
    except RuntimeError:
        pass

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported policy checkpoint: {checkpoint_path}")
    model_state = checkpoint["model_state_dict"]
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in model_state.items()
        if key.startswith("actor.")
    }
    if not actor_state:
        raise RuntimeError(f"Checkpoint has no actor.* weights: {checkpoint_path}")
    return build_mlp_from_actor_state(actor_state).to(device).eval()


def export_actor(
    checkpoint_path: str | Path,
    output_path: str | Path,
    obs_dim: int,
    *,
    onnx_path: str | Path | None = None,
) -> None:
    actor = load_actor_from_checkpoint(checkpoint_path, "cpu")
    example = torch.zeros(1, obs_dim, dtype=torch.float32)
    traced = torch.jit.trace(actor, example)
    traced.save(str(output_path))
    if onnx_path is not None:
        torch.onnx.export(
            actor,
            example,
            str(onnx_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
