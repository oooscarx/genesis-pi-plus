#!/usr/bin/env python3
"""Train pi_plus kick policy in chunks and stop when stability collapses."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis_pi_plus.config import load_config
from genesis_pi_plus.kick_env import PiPlusKickEnv, default_kick_env_paths


def main() -> None:
    args = parse_args()
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as exc:
        raise SystemExit("rsl-rl-lib is not installed. Run `uv sync`.") from exc

    paths = default_kick_env_paths(ROOT)
    train_cfg = load_config(paths.train_cfg)
    if args.num_envs is not None:
        train_cfg["num_envs"] = args.num_envs

    env = PiPlusKickEnv(
        paths,
        num_envs=train_cfg["num_envs"],
        device=args.device or train_cfg.get("device", "cuda"),
        backend=args.backend,
        headless=True,
        build_scene=True,
    )
    runner_cfg = train_cfg["rsl_rl"]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=str(log_dir), device=str(env.device))

    best_score = -1.0e9
    completed = 0
    first_chunk = True
    while completed < args.iterations:
        chunk = min(args.chunk_iterations, args.iterations - completed)
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=first_chunk)
        materialize_env_buffers(env)
        first_chunk = False
        completed += chunk

        metrics = evaluate_runner_policy(runner, env, args.eval_steps)
        score = metrics["ball_contact_mean"] * 5.0 - metrics["foot_ball_distance_m_mean"] - metrics["fall_rate_mean"] * 10.0
        print(
            "[guard] "
            f"iter={completed} score={score:.4f} "
            f"fall_rate_mean={metrics['fall_rate_mean']:.4f} "
            f"contact_mean={metrics['ball_contact_mean']:.4f} "
            f"foot_ball_distance_m={metrics['foot_ball_distance_m_mean']:.4f}"
        )

        checkpoint = log_dir / f"guard_iter_{completed}.pt"
        runner.save(str(checkpoint), infos={"guard_metrics": metrics})
        if score > best_score and metrics["fall_rate_mean"] <= args.max_fall_rate:
            best_score = score
            runner.save(str(log_dir / "guard_best.pt"), infos={"guard_metrics": metrics})

        if metrics["fall_rate_mean"] > args.max_fall_rate:
            print(f"[guard] stopping: fall_rate_mean {metrics['fall_rate_mean']:.4f} > {args.max_fall_rate:.4f}")
            break
        if metrics["foot_ball_distance_m_mean"] > args.max_foot_ball_distance:
            print(
                "[guard] stopping: "
                f"foot_ball_distance_m {metrics['foot_ball_distance_m_mean']:.4f} > {args.max_foot_ball_distance:.4f}"
            )
            break


@torch.no_grad()
def evaluate_runner_policy(runner, env: PiPlusKickEnv, n_steps: int) -> dict[str, float]:
    runner.eval_mode()
    obs, _ = env.reset()
    fall_sum = torch.zeros((), device=env.device)
    contact_sum = torch.zeros((), device=env.device)
    distance_sum = torch.zeros((), device=env.device)
    reward_sum = torch.zeros((), device=env.device)
    for _ in range(n_steps):
        obs_for_policy = runner.obs_normalizer(obs.to(runner.device))
        actions = runner.alg.policy.act_inference(obs_for_policy).to(env.device)
        obs, rewards, _done, extras = env.step(actions)
        reward_sum += rewards.mean()
        fall_sum += extras["log"].get("episode/fall_rate", torch.zeros((), device=env.device))
        contact_sum += extras["log"].get("reward/ball_contact", torch.zeros((), device=env.device))
        distance_sum += extras["log"].get("metric/foot_ball_distance_m", torch.zeros((), device=env.device))
    runner.train_mode()
    materialize_env_buffers(env)
    denom = max(n_steps, 1)
    return {
        "mean_reward": float(reward_sum / denom),
        "fall_rate_mean": float(fall_sum / denom),
        "ball_contact_mean": float(contact_sum / denom),
        "foot_ball_distance_m_mean": float(distance_sum / denom),
    }


def materialize_env_buffers(env: PiPlusKickEnv) -> None:
    """Turn tensors created under torch.inference_mode back into normal tensors."""
    for name in (
        "actions",
        "prev_actions",
        "prev_delta",
        "last_torque",
        "ball_pos",
        "prev_ball_pos",
        "target_pos",
        "desired_ball_speed",
        "episode_length_buf",
    ):
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor):
            setattr(env, name, value.clone())
    baseline = getattr(env, "baseline", None)
    if baseline is not None:
        for name in ("obs_history", "last_action_isaac"):
            value = getattr(baseline, name, None)
            if isinstance(value, torch.Tensor):
                setattr(baseline, name, value.clone())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded pi_plus kick training.")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--chunk-iterations", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=125)
    parser.add_argument("--max-fall-rate", type=float, default=0.15)
    parser.add_argument("--max-foot-ball-distance", type=float, default=1.0)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default="runs/pi_plus_kick_guarded")
    return parser.parse_args()


if __name__ == "__main__":
    main()
