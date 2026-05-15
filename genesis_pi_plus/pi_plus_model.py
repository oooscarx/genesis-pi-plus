"""Robot metadata container for pi_plus."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PiPlusModelInfo:
    """Static robot metadata used by controllers and test scripts."""

    joint_names: list[str] = field(default_factory=list)
    default_joint_pos: dict[str, float] = field(default_factory=dict)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    pd_kp: dict[str, float] = field(default_factory=dict)
    pd_kd: dict[str, float] = field(default_factory=dict)
    foot_link_names: list[str] = field(default_factory=list)
    base_link_name: str | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "PiPlusModelInfo":
        robot = cfg.get("robot", {})
        return cls(
            joint_names=list(robot.get("joint_names") or []),
            default_joint_pos=dict(robot.get("default_joint_pos") or {}),
            joint_limits=_parse_joint_limits(robot.get("joint_limits") or {}),
            pd_kp=dict(robot.get("pd_kp") or {}),
            pd_kd=dict(robot.get("pd_kd") or {}),
            foot_link_names=list(robot.get("foot_link_names") or []),
            base_link_name=robot.get("base_link_name"),
        )


def _parse_joint_limits(raw: dict) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for name, value in raw.items():
        if value is None:
            continue
        if isinstance(value, dict):
            lo = value.get("lower")
            hi = value.get("upper")
        else:
            lo, hi = value
        limits[name] = (float(lo), float(hi))
    return limits
