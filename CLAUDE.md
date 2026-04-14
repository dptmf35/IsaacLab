# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Isaac Lab is a GPU-accelerated robotics research framework built on NVIDIA Isaac Sim. It supports reinforcement learning, imitation learning, and motion planning with physics-accurate simulation. Python 3.11, Isaac Sim 4.5/5.0/5.1.

## Common Commands

All scripts must be run through the `isaaclab.sh` launcher to ensure correct Isaac Sim Python environment resolution:

```bash
# Install extensions and RL extras
./isaaclab.sh -i

# Run a Python script
./isaaclab.sh -p scripts/demos/arms.py

# Run pre-commit hooks (ruff lint+format, codespell, YAML/TOML checks)
./isaaclab.sh -f

# Run all tests
./isaaclab.sh -t

# Run a specific test directory
./isaaclab.sh -p -m pytest source/isaaclab/test/controllers

# Discover all tests without running
./isaaclab.sh -p tools/run_all_tests.py --discover_only

# Build Sphinx docs
./isaaclab.sh -d
```

### RL Training & Evaluation

```bash
# Train with RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <TASK_ID> --headless

# Play/evaluate a checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task <TASK_ID>-Play-v0 --checkpoint <path/to/model_xxx.pt> --num_envs 1
```

Supported RL frameworks: RSL-RL, SKRL, RL-Games, Stable Baselines3. Training scripts are under `scripts/reinforcement_learning/<framework>/`.

## Architecture

### Extension Layout (`source/`)

| Extension | Purpose |
|-----------|---------|
| `isaaclab` | Core framework: assets, envs, managers, sensors, sim, terrains, controllers |
| `isaaclab_tasks` | Task environments organized as `manager_based/` and `direct/` |
| `isaaclab_assets` | Robot and object asset configs |
| `isaaclab_rl` | RL framework wrappers (RSL-RL, SKRL, etc.) |
| `isaaclab_mimic` | Imitation learning / data generation |
| `isaaclab_contrib` | Community contributions |

Tests are colocated under each extension's `test/` directory (`test_*.py` naming).

### Two Environment Paradigms

**Manager-Based** (`source/isaaclab/isaaclab/envs/manager_based_rl_env.py`):
- Composable via manager classes: `ActionManager`, `ObservationManager`, `RewardManager`, `TerminationManager`, `CommandManager`, `EventManager`, `CurriculumManager`
- Configs live in `source/isaaclab_tasks/isaaclab_tasks/manager_based/<domain>/<task>/`
- Robot-specific overrides in `config/<robot>/`, agent configs in `config/<robot>/agents/`

**Direct** (`source/isaaclab/isaaclab/envs/direct_rl_env.py`):
- Subclass and override `_setup_scene`, `_pre_physics_step`, `_get_observations`, `_get_rewards`, `_get_dones`
- Configs live in `source/isaaclab_tasks/isaaclab_tasks/direct/<task>/`

### Key Framework Modules

- `isaaclab/sim/` — Isaac Sim wrappers (spawners, schemas, converters, utils)
- `isaaclab/assets/` — Articulation, RigidObject, DeformableObject
- `isaaclab/sensors/` — Camera, RayCaster, ContactSensor, IMU, FrameTransformer
- `isaaclab/terrains/` — Procedural terrain generation
- `isaaclab/controllers/` — Differential IK, OSC, joint controllers
- `isaaclab/scene/` — InteractiveScene (scene graph management)

### Config System (`@configclass`)

All environment, asset, sensor, and agent configs use `@configclass` (a dataclass wrapper). Configs compose via inheritance; override fields in `__post_init__` after calling `super().__post_init__()`. This is the primary extension mechanism — do not instantiate config objects with keyword args; set attributes instead.

### Custom Tasks in This Repo

**Manipulation (UR10/UR10e deploy tasks)** — `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/`:
- `reach/` — `Isaac-Deploy-Reach-UR10e-v0` (RSL-RL PPO, non-recurrent); also has `-ROS-Inference-v0` variant for real-robot deployment
- `gear_assembly/` — `Isaac-Deploy-GearAssembly-UR10e-2F140-v0` / `2F85-v0` (RSL-RL Recurrent PPO with LSTM); also has `-ROS-Inference-v0` variants that expose `obs_order`, `action_space`, noise parameters for the ROS inference node
- `mdp/` — Shared rewards, observations, terminations, events, and `noise_models.py` (observation noise for sim-to-real)

**Navigation (Spot)** — `source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/spot/`:
- Hierarchical controller: high-level navigation policy → low-level locomotion policy → joint actions
- Standalone app: `projects/spot_standalone_nav/` (runs via Isaac Sim `python.sh`, not `isaaclab.sh`)

### Project-Level Apps (`projects/`)

Self-contained applications with their own `run.sh` launchers. `projects/spot_standalone_nav/` is the primary example — see its `README.md` for setup.

## Code Style

- **Ruff** line length 120; isort with custom section ordering for `isaaclab*` packages
- Import order: stdlib → third-party → omniverse extensions → isaaclab → isaaclab_contrib/rl/mimic/tasks/assets → first-party
- `snake_case` modules/functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Run `./isaaclab.sh -f` before committing

## Adding a New Task

1. Create task directory under `source/isaaclab_tasks/isaaclab_tasks/manager_based/<domain>/<task>/`
2. Define `<task>_env_cfg.py` with a `@configclass` subclass of `ManagerBasedRLEnvCfg`; override in `__post_init__`
3. Add robot-specific override in `config/<robot>/joint_pos_env_cfg.py`
4. Add agent config in `config/<robot>/agents/rsl_rl_ppo_cfg.py`
5. Register train (`-v0`) and play (`-Play-v0`) gym IDs in `config/<robot>/__init__.py`
