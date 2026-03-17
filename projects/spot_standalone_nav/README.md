# Spot Standalone Navigation Project

Standalone-style Isaac Sim app for running Spot with:
- High-level navigation policy (trained in IsaacLab)
- Low-level walking policy (trained/exported in IsaacLab)

This project is intended to run with Isaac Sim `python.sh` (not `isaaclab.sh`).

## Project Layout
- `applications/spot_navigation_app.py`: main standalone app
- `run.sh`: convenience launcher
- `configs/env.example`: environment variable template

## Prerequisites
- Isaac Sim installed (with `python.sh`)
- IsaacLab repository available (for task/env Python modules)
- Trained checkpoints:
  - High-level nav: `model_XXXX.pt` or `exported/policy.pt`
  - Low-level walk: `exported/policy.pt`

## Quick Start
1. Copy env template and edit paths:
```bash
cp projects/spot_standalone_nav/configs/env.example projects/spot_standalone_nav/configs/env.local
```

2. Load env:
```bash
source projects/spot_standalone_nav/configs/env.local
```

3. Run:
```bash
projects/spot_standalone_nav/run.sh \
  --task Isaac-Navigation-FullWarehouse-Spot-Play-v0 \
  --checkpoint "$SPOT_NAV_CKPT" \
  --checkpoint-type rsl \
  --low-level-policy "$SPOT_LL_POLICY" \
  --num_envs 1 \
  --disable-timeout \
  --waypoint-step 3.5
```

## Runtime Commands
In app console:
- `x y yaw`: set goal (auto-waypoint segmentation enabled)
- `c`: cancel current goal
- `stop`: stop robot
- `quit`: exit app

## Notes
- If `--checkpoint-type auto`, the app tries JIT first, then raw RSL checkpoint.
- Full warehouse USD path can be overridden via:
  - `ISAACLAB_SPOT_FULL_WAREHOUSE_USD_PATH`
