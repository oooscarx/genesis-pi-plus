# genesis_pi_plus

`genesis_pi_plus` is a standalone Git repository for migrating the `pi_plus` robot assets and control checks from `AMP_TK` / IsaacLab / MuJoCo into Genesis. The initial phase focused on Genesis adaptation validation. The current tree also includes a residual kick-training scaffold using Genesis and rsl-rl, still with no Isaac Lab dependency.

The Genesis runtime asset is self-contained under `assets/pi_plus/`, with STL meshes tracked through Git LFS. Normal load/train/play runs do not require an adjacent `AMP_TK` checkout.

## Local Setup on Mac

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and sync the environment:

```bash
cd genesis_pi_plus
uv sync
uv run python scripts/inspect_amp_tk.py
```

On a Mac without CUDA, the scripts will fall back from `sim.backend: cuda` to CPU for smoke testing. You can also force a backend:

```bash
GENESIS_BACKEND=cpu uv run python scripts/test_genesis_load_pi_plus.py
```

For locked installs:

```bash
uv sync --frozen
```

Run smoke scripts:

```bash
uv run python scripts/test_genesis_load_pi_plus.py
uv run python scripts/test_pi_plus_pd_stand.py
uv run python scripts/test_pi_plus_ball_contact.py
```

To run the exported MuJoCo-validated locomotion policy in Genesis:

```bash
GENESIS_BACKEND=metal uv run python scripts/test_pi_plus_policy_walk.py --duration 5
```

The default policy path is `policies/model_40000.pt`, tracked with Git LFS. This is a rollout smoke test for migration validation, not training.

The viewer ground plane color, tile size, and camera are configured in `configs/pi_plus_genesis.yaml` under `scene`.

## GitHub Upload

```bash
git remote add origin git@github.com:<your-org-or-user>/genesis_pi_plus.git
git push -u origin main
```

## Headless Ubuntu Training Host

Install system packages first:

```bash
apt update
apt install -y git git-lfs curl build-essential cmake ninja-build \
  libgl1 libglib2.0-0 libegl1 libegl-dev libsm6 libxext6 libxrender1
git lfs install
```

```bash
git clone git@github.com:<your-org-or-user>/genesis_pi_plus.git
cd genesis_pi_plus
git lfs pull
uv sync --frozen
uv run python scripts/test_genesis_load_pi_plus.py
```

For RTX 5090 / Blackwell hosts, check that the installed PyTorch build supports `sm_120`. If it does not, install a PyTorch wheel/nightly build that supports the GPU before running Genesis workloads.

## Docker

Build:

```bash
docker build -t genesis-pi-plus:cuda128 .
```

Run:

```bash
docker run --rm -it --gpus all --ipc=host -v $PWD:/workspace genesis-pi-plus:cuda128
```

The default container command runs headless and does not open a viewer.

## Filling `configs/pi_plus_genesis.yaml`

The default asset path is already vendored in this repository:

```yaml
robot:
  asset_file: assets/pi_plus/pi_plus.xml
```

When refreshing from a newer AMP_TK release, start with:

```bash
uv run python scripts/inspect_amp_tk.py
```

Then open `reports/pi_plus_asset_report.md` and verify:

- `robot.asset_file`: should point to `assets/pi_plus/pi_plus.xml` unless you are deliberately testing an external vendor asset.
- `robot.joint_names`: must match the action/control order used for Genesis targets.
- `robot.default_joint_pos`, `robot.pd_kp`, `robot.pd_kd`, and `robot.action_scale`: currently seeded from `sim2sim_pi_plus.py`; re-check after Genesis joint introspection.
- `robot.foot_link_names`: still TODO until foot/contact link names are verified in Genesis.
- `sim.sim_dt` and `sim.control_dt`: currently seeded from MuJoCo sim2sim values.

Any field that cannot be verified should stay `null`, empty, or explicitly marked TODO.

## Safety Before Real Robot Deployment

Before any real pi_plus deployment, add and test a safety layer with joint position limits, velocity limits, IMU roll/pitch protection, emergency stop, and a kickable-area check before kicking the ball.

## Kick Training

The kick training stack is scaffolded around Genesis and `rsl-rl-lib`. The policy action is a 20D residual joint-angle target added to a frozen baseline target. By default, `configs/pi_plus_kick_train.yaml` uses `baseline.mode: locomotion_policy` with `policies/model_40000.pt`; set it to `default_pose` only for debugging the residual controller without the locomotion actor.

The default kick config starts as an approach/contact curriculum: the ball is placed near the front-right foot, foot-ball proximity/contact rewards are strong, and target-distance rewards are weak. Ball rewards are gated by upright/base-height stability so a falling policy cannot get useful kick reward. Episodes terminate early when foot-ball distance escapes the curriculum region. First make `metric/foot_ball_distance_m` decrease and `reward/ball_contact` become nonzero while keeping `episode/fall_rate` low; only then increase target/kick-speed rewards.

```bash
uv sync --frozen
uv run python scripts/train_pi_plus_kick.py --num-envs 1024 --backend cuda --device cuda
uv run python scripts/train_pi_plus_kick_guarded.py --num-envs 1024 --iterations 10000 --backend cuda --device cuda
uv run python scripts/test_pi_plus_kick_components.py
uv run python scripts/eval_pi_plus_kick.py --checkpoint runs/pi_plus_kick/model_*.pt --num-envs 256
uv run python scripts/play_pi_plus_kick.py --checkpoint runs/pi_plus_kick/model_*.pt --backend metal --device mps --viewer
uv run python scripts/export_pi_plus_kick_policy.py --checkpoint runs/pi_plus_kick/model_*.pt --output exports/pi_plus_kick_policy.pt
```

Configuration lives in `configs/pi_plus_kick_train.yaml`, `configs/pi_plus_kick_rewards.yaml`, and `configs/pi_plus_domain_randomization.yaml`. RTX 5090 training hosts should use CUDA 12.8+ and a PyTorch build with Blackwell / `sm_120` support.

### TensorBoard

Training logs are written under the `--log-dir` path, for example `runs/pi_plus_kick_256`. On a remote training host:

```bash
uv run tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

Then forward the port from your local machine:

```bash
ssh -L 6006:127.0.0.1:6006 root@<remote-host>
```

Open `http://127.0.0.1:6006` locally. If the cloud provider exposes a web terminal/proxy, you can also open port `6006` through that panel.
