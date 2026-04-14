# Spot Standalone Navigation Guide

## Added File
- `scripts/tutorials/03_envs/spot_navigation_in_usd_standalone.py`

## What This Script Does
- Runs Spot navigation in **standalone Isaac Sim** (not `rsl_rl/play.py`).
- Loads high-level navigation policy from exported JIT `policy.pt`.
- Uses low-level Spot walking policy (`policy.pt`) for hierarchical control.
- Initializes robot at `(0, 0, 0)` with zero velocity.
- Keeps robot standing still until first goal command is entered.
- Supports runtime commands while sim is running:
  - `x y yaw`: set new goal immediately
  - `stop`: stop and hold
  - `quit`: exit simulation

## Required Checkpoints
- High-level nav JIT:
  - `logs/rsl_rl/spot_navigation/<run>/exported/policy.pt`
- Low-level Spot JIT:
  - `logs/rsl_rl/spot_flat/<run>/exported/policy.pt`

Do **not** use `model_XXXX.pt` as `--checkpoint` in this standalone script.

## Run Command
```bash
NAV_JIT=logs/rsl_rl/spot_navigation/<run>/exported/policy.pt
SPOT_LL=logs/rsl_rl/spot_flat/<run>/exported/policy.pt

./isaaclab.sh -p scripts/tutorials/03_envs/spot_navigation_in_usd_standalone.py \
  --checkpoint "$NAV_JIT" \
  --low-level-policy "$SPOT_LL" \
  --num_envs 1
```

## Runtime Usage
- Wait for prompt: `goal>`
- Enter first goal: e.g. `1.0 1.0 -1.57`
- Change goal anytime with another `x y yaw`
- Emergency stop: `stop`
- Exit: `quit`
