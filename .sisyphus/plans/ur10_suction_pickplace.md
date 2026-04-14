# Work Plan: UR10 Suction Pick-and-Place Task

## Context

### Original Request
Create a new RL task for UR10 with suction gripper to pick and place a single box.
Binary suction action (on/off), randomized pick and place positions, using existing
UR10 suction robot assets (`UR10_LONG_SUCTION_CFG` or `UR10_SHORT_SUCTION_CFG`).

### Interview Summary
- **Suction modeling**: Binary action (on/off) - adds 1 dimension to action space
- **Object count**: 1 box, pick and place
- **Goal position**: Randomized (both pick and place positions randomized)
- **Robot asset**: UR10 with suction (`UR10_LONG_SUCTION_CFG` / `UR10_SHORT_SUCTION_CFG`)
- **Reward design**: Hybrid (C) - phase-based dense rewards + success bonus
- **RL framework**: RSL-RL PPO, non-recurrent first (LSTM later)

### Research Findings (Codebase Patterns)

**SurfaceGripper pattern** (from `stack/config/ur10_gripper/`):
- `SurfaceGripperCfg` added as scene entity `surface_gripper`
- `SurfaceGripperBinaryActionCfg` used as `gripper_action` in ActionsCfg
- **CRITICAL**: Suction grippers currently require CPU simulation (`self.device = "cpu"`)
- `SurfaceGripper` prim path: `{ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper`
- Parameters: `max_grip_distance=0.0075`, `shear_force_limit=5000.0`, `coaxial_force_limit=5000.0`, `retry_interval=0.05`

**Deploy task pattern** (from `deploy/reach/`):
- Base env cfg defines Scene, Actions, Observations, Events, Rewards, Terminations
- Robot-specific config in `config/ur_10e/joint_pos_env_cfg.py` overrides `__post_init__`
- Agent config in `config/ur_10e/agents/rsl_rl_ppo_cfg.py`
- Gym registration in `config/ur_10e/__init__.py` with train (`-v0`) and play (`-Play-v0`) IDs
- Uses `FrameTransformerCfg` for ee_frame tracking

**RigidObject pattern** (from `gear_assembly/` and `stack/`):
- Box USD available: `{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd` (from stack task)
- `RigidObjectCfg` with rigid body properties, collision properties
- Randomization via `reset_root_state_uniform` EventTerm

**Existing MDP functions available** (`deploy/mdp/`):
- `joint_pos`, `joint_vel` observation terms
- `action_rate_l2`, `action_l2` reward terms
- `time_out` termination term
- `reset_joints_by_offset`, `reset_root_state_uniform` event terms
- `randomize_actuator_gains`, `randomize_joint_parameters` event terms
- `RelativeJointPositionActionCfg` for arm action

---

## Work Objectives

### Core Objective
Implement a manager-based RL environment for UR10 suction pick-and-place with phased dense rewards.

### Deliverables
1. New task directory with env cfg, MDP modules, robot config, agent config, and gym registration
2. Trainable with RSL-RL PPO via standard `train.py` script
3. Playable with standard `play.py` script

### Definition of Done
- [ ] `Isaac-Deploy-PickPlace-UR10-Suction-v0` gym env registered and runnable
- [ ] `Isaac-Deploy-PickPlace-UR10-Suction-Play-v0` gym env registered
- [ ] Training converges (reward increases over 500+ iterations)
- [ ] Box successfully picked and placed in visualization

---

## Guardrails

### Must Have
- Binary suction action (1D: on/off) via `SurfaceGripperBinaryActionCfg`
- Phased dense reward (approach -> pick -> place)
- Randomized pick position AND goal position
- Joint position observations, EE pose, box pose, goal pose, suction state
- Timeout and box-drop termination conditions
- Success termination (box near goal)
- CPU simulation mode (suction gripper requirement)

### Must NOT Have
- Camera/image observations (state-only for v1)
- Multiple boxes
- LSTM/recurrent policy (start with MLP)
- ROS inference variant (future work)
- Domain randomization beyond basic joint noise (keep simple for v1)

---

## Directory Structure

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/pick_place_suction/
    __init__.py                          # Empty, re-exports config
    pick_place_suction_env_cfg.py        # Base env cfg (scene, MDP managers)
    mdp/
        __init__.py                      # Re-exports all MDP modules
        observations.py                  # Custom obs: box_pos, goal_pos, suction_state, ee_pos
        rewards.py                       # Phased rewards: approach, pick, place, success bonus
        terminations.py                  # box_dropped, box_at_goal (success)
        events.py                        # reset_box_pose, reset_goal_pose
    config/
        __init__.py                      # Empty
        ur_10/
            __init__.py                  # Gym registration (train + play IDs)
            joint_pos_env_cfg.py         # UR10 Long Suction specific overrides
            agents/
                __init__.py              # Empty
                rsl_rl_ppo_cfg.py        # RSL-RL PPO hyperparameters
```

**Also modify:**
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/__init__.py` - add `from .pick_place_suction import *`

---

## Task Flow and Dependencies

```
Task 1: Base env cfg (scene + MDP structure)
    |
    +---> Task 2: Custom MDP modules (obs, rewards, terminations, events)
    |         |
    |         v
    +---> Task 3: Robot-specific config (UR10 suction overrides)
              |
              v
          Task 4: Agent config (RSL-RL PPO)
              |
              v
          Task 5: Gym registration + __init__.py wiring
              |
              v
          Task 6: Smoke test (env creation + short training)
```

---

## Detailed Tasks

### Task 1: Base Environment Config (`pick_place_suction_env_cfg.py`)

**What to implement:**

1. **SceneCfg** (extends `InteractiveSceneCfg`):
   - `ground`: GroundPlane at (0, 0, -1.05)
   - `table`: Stand USD (same as reach task, scale 2.0)
   - `robot`: `ArticulationCfg = MISSING` (filled by robot config)
   - `surface_gripper`: `SurfaceGripperCfg = MISSING` (filled by robot config)
   - `box`: `RigidObjectCfg` using `{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd`
     - `rigid_props`: disable_gravity=False, solver_position_iteration_count=16
     - `init_state`: pos=[0.5, 0.0, 0.02], rot=[1,0,0,0]
   - `ee_frame`: `FrameTransformerCfg = MISSING` (filled by robot config)
   - `goal_marker`: `AssetBaseCfg` visual-only sphere/cube at goal position for debug visualization
   - `light`: DomeLight
   - `replicate_physics = False` (required for suction gripper)

2. **CommandsCfg**:
   - `goal_pose`: `UniformPoseCommandCfg` for the place goal position
     - Resampling tied to episode reset
     - Ranges: x=(0.3, 0.7), y=(-0.25, 0.25), z=(0.02, 0.02) (on table surface)
     - No rotation command needed (just position)

3. **ActionsCfg**:
   - `arm_action`: `ActionTerm = MISSING`
   - `gripper_action`: `SurfaceGripperBinaryActionCfg` (asset_name="surface_gripper", open=-1.0, close=1.0)

4. **ObservationsCfg**: -> defined in mdp/observations.py, referenced here
5. **RewardsCfg**: -> defined in mdp/rewards.py, referenced here
6. **TerminationsCfg**: -> defined in mdp/terminations.py, referenced here
7. **EventCfg**: -> defined in mdp/events.py, referenced here

8. **PickPlaceSuctionEnvCfg** (extends `ManagerBasedRLEnvCfg`):
   - `decimation = 4`
   - `episode_length_s = 15.0` (longer than reach: pick+place needs more time)
   - `sim.dt = 1.0 / 120.0`
   - `device = "cpu"` (suction gripper requirement)

**Acceptance criteria:**
- Class compiles without import errors
- MISSING fields are properly typed for downstream override

---

### Task 2: Custom MDP Modules

#### Task 2a: Observations (`mdp/observations.py`)

Implement the following observation terms:

| Term | Type | Shape | Source |
|------|------|-------|--------|
| `joint_pos` | Reuse | (6,) | `mdp.joint_pos` from deploy/mdp |
| `joint_vel` | Reuse | (6,) | `mdp.joint_vel` from deploy/mdp |
| `ee_pos_w` | New func | (3,) | FrameTransformer ee_frame target_pos_source[:, 0] - env_origins |
| `ee_quat_w` | New func | (4,) | FrameTransformer ee_frame target_quat_source[:, 0] |
| `box_pos_w` | New func | (3,) | box RigidObject root_pos_w - env_origins |
| `box_quat_w` | New func | (4,) | box RigidObject root_quat_w |
| `goal_pos` | Reuse | (3,) | `mdp.generated_commands` command_name="goal_pose" (position part) |
| `suction_state` | New func | (1,) | surface_gripper.data.is_gripping (bool -> float) |

**Total observation dim**: 6 + 6 + 3 + 4 + 3 + 4 + 3 + 1 = **30**

**Implementation notes:**
- `ee_pos_w`, `box_pos_w`: subtract `env.scene.env_origins` to get env-local positions
- `suction_state`: Check `SurfaceGripper` data API for gripping state. If no direct API, track via action history (last gripper command).
- For `goal_pos`: Use `generated_commands` with the command name, extract position only (first 3 dims). May need a wrapper if command includes orientation.

**Acceptance criteria:**
- Each observation function returns correct tensor shape
- Observation concatenation produces (num_envs, 30) tensor

#### Task 2b: Rewards (`mdp/rewards.py`)

Implement phased dense reward functions:

**Phase 1 - Approach (EE -> Box)**
```python
class approach_box_reward(ManagerTermBase):
    """Dense reward for moving EE toward the box.
    reward = exp(-alpha * ||ee_pos - box_pos||)
    Only active when suction is OFF.
    Weight: 1.0
    """
```

**Phase 2 - Lift (Box attached, move up)**
```python
class lift_box_reward(ManagerTermBase):
    """Reward for lifting the box after successful grasp.
    reward = clip(box_z - table_height, 0, lift_target) / lift_target
    Only active when suction is ON and box is grasped.
    Weight: 2.0
    """
```

**Phase 3 - Place (Box -> Goal)**
```python
class place_box_reward(ManagerTermBase):
    """Dense reward for moving box toward goal position.
    reward = exp(-beta * ||box_pos - goal_pos||)
    Only active when suction is ON and box is lifted.
    Weight: 3.0
    """
```

**Auxiliary rewards:**
```python
# Reuse from deploy/mdp
action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
action = RewTerm(func=mdp.action_l2, weight=-0.005)

# Success bonus
class success_bonus(ManagerTermBase):
    """Large sparse reward when box is placed at goal.
    reward = 10.0 if ||box_pos - goal_pos|| < threshold (0.05m) AND suction OFF
    Weight: 1.0
    """

# Grasp reward
class grasp_reward(ManagerTermBase):
    """Small reward when suction activates near box.
    reward = 1.0 if suction ON and ||ee_pos - box_pos|| < grasp_threshold
    Weight: 0.5
    """
```

**Acceptance criteria:**
- Rewards are non-zero and correctly phased during training
- Total reward scale allows learning (not too sparse, not too noisy)

#### Task 2c: Terminations (`mdp/terminations.py`)

```python
class box_dropped(ManagerTermBase):
    """Reset if box falls below table.
    Condition: box_z < -0.05 (below table surface)
    """

class box_at_goal(ManagerTermBase):
    """Success termination.
    Condition: ||box_pos_xy - goal_pos_xy|| < 0.05 AND box on table AND suction OFF
    """

# Also reuse: time_out from deploy/mdp
```

**Acceptance criteria:**
- Episodes terminate on success (box placed correctly)
- Episodes terminate when box drops off table
- Episodes terminate on timeout

#### Task 2d: Events (`mdp/events.py`)

```python
class reset_box_random(ManagerTermBase):
    """Randomize box position on table at episode reset.
    x: (0.35, 0.65), y: (-0.2, 0.2), z: 0.02 (on table surface)
    """
    # Can use mdp.reset_root_state_uniform with appropriate params

# Goal position randomization is handled by CommandsCfg (UniformPoseCommandCfg)

# Robot reset: reuse reset_joints_by_offset from deploy/mdp
```

**Acceptance criteria:**
- Box appears at random valid positions each episode
- Goal position varies each episode
- No overlap between box initial position and goal position (add min separation check)

#### Task 2e: MDP `__init__.py`

```python
from isaaclab_tasks.manager_based.manipulation.deploy.mdp import *  # noqa
from .observations import *  # noqa
from .rewards import *  # noqa
from .terminations import *  # noqa
from .events import *  # noqa
```

---

### Task 3: Robot-Specific Config (`config/ur_10/joint_pos_env_cfg.py`)

**UR10LongSuctionPickPlaceEnvCfg** (extends `PickPlaceSuctionEnvCfg`):

```python
def __post_init__(self):
    super().__post_init__()

    # CPU required for suction
    self.device = "cpu"

    # Robot
    self.scene.robot = UR10_LONG_SUCTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Surface gripper
    self.scene.surface_gripper = SurfaceGripperCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper",
        max_grip_distance=0.0075,
        shear_force_limit=5000.0,
        coaxial_force_limit=5000.0,
        retry_interval=0.05,
    )

    # EE frame (offset for long suction: 0.22 along x)
    self.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link",
            name="end_effector",
            offset=OffsetCfg(pos=[0.22, 0.0, 0.0]),
        )],
    )

    # Arm action: relative joint position
    self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.0625, use_zero_offset=True
    )

    # Gripper action
    self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
        asset_name="surface_gripper",
        open_command=-1.0,
        close_command=1.0,
    )
```

Also create `UR10LongSuctionPickPlaceEnvCfg_PLAY` with:
- `num_envs = 50`
- `enable_corruption = False`

**Acceptance criteria:**
- Robot + suction gripper initialized correctly in scene
- Action space = 6 (joints) + 1 (suction) = 7 dimensions
- Observation space = 30 dimensions

---

### Task 4: Agent Config (`config/ur_10/agents/rsl_rl_ppo_cfg.py`)

```python
@configclass
class PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 512       # same as reach
    max_iterations = 3000         # more iterations for harder task
    save_interval = 100
    experiment_name = "pick_place_ur10_suction"
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],   # slightly larger for harder task
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,        # small entropy bonus for exploration
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=3.0e-4,      # slightly lower for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
```

**Acceptance criteria:**
- PPO config loads without error
- `num_envs` reasonable for CPU mode (start with 256-512, NOT 4096)

---

### Task 5: Gym Registration and Wiring

#### `config/ur_10/__init__.py`
```python
gym.register(
    id="Isaac-Deploy-PickPlace-UR10-Suction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10LongSuctionPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-PickPlace-UR10-Suction-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10LongSuctionPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
)
```

#### Update `deploy/__init__.py`
Add: `from .pick_place_suction import *  # noqa: F401, F403`

#### All intermediate `__init__.py` files
- `pick_place_suction/__init__.py`: `from .config import *`
- `pick_place_suction/config/__init__.py`: empty or import ur_10
- `pick_place_suction/mdp/__init__.py`: import all sub-modules

**Acceptance criteria:**
- `gym.make("Isaac-Deploy-PickPlace-UR10-Suction-v0")` succeeds
- Task appears in `isaaclab` task list

---

### Task 6: Smoke Test

1. **Environment creation test:**
   ```bash
   ./isaaclab.sh -p -c "import gymnasium as gym; import isaaclab_tasks; env = gym.make('Isaac-Deploy-PickPlace-UR10-Suction-v0', num_envs=4); print('OK:', env.observation_space, env.action_space)"
   ```

2. **Short training test:**
   ```bash
   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task Isaac-Deploy-PickPlace-UR10-Suction-v0 --headless --num_envs 64 --max_iterations 50
   ```

3. **Verify:**
   - No crashes during 50 iterations
   - Reward values change (not constant)
   - Action space is 7D (6 joints + 1 suction)
   - Observation space is 30D

**Acceptance criteria:**
- Training runs without error for 50 iterations
- Tensorboard shows non-trivial reward curves

---

## Key Technical Decisions

### 1. Suction Modeling
Using Isaac Sim's built-in `SurfaceGripper` via `SurfaceGripperCfg` + `SurfaceGripperBinaryActionCfg`.
This creates a physics-based constraint when activated near an object surface.

**Trade-off**: Requires CPU simulation (no GPU acceleration), limiting parallelism.
Mitigate by using fewer environments (256-512 instead of 4096).

### 2. Reward Design (Hybrid/Phased)

| Phase | Condition | Reward | Weight |
|-------|-----------|--------|--------|
| Approach | suction OFF | exp(-10 * dist(ee, box)) | 1.0 |
| Grasp | suction ON near box | +1.0 (one-time) | 0.5 |
| Lift | box grasped, box_z > threshold | (box_z - table_z) / lift_height | 2.0 |
| Place | box lifted | exp(-10 * dist(box, goal)) | 3.0 |
| Success | box at goal, suction OFF | +10.0 (sparse) | 1.0 |
| Action penalty | always | -||action||^2, -||action_rate||^2 | -0.005 |

Weights increase for later phases to encourage completion of the full task.

### 3. Observation Space

All observations in env-local frame (subtract `env_origins`). No normalization at env level --
handled by RSL-RL's `empirical_normalization`.

### 4. Goal as Command vs Static

Using `UniformPoseCommandCfg` for goal position allows:
- Automatic resampling on reset
- Debug visualization for free
- Standard `generated_commands` observation integration

### 5. Box Spawn and Goal Separation

Enforce minimum distance between box initial position and goal position (0.15m) in the
event randomization to ensure non-trivial tasks.

---

## Commit Strategy

| Commit | Content |
|--------|---------|
| 1 | Scaffold: directory structure, empty `__init__.py` files, base env cfg |
| 2 | MDP modules: observations, rewards, terminations, events |
| 3 | Robot config + agent config + gym registration |
| 4 | Integration: wiring, import fixes, smoke test verification |

---

## Success Criteria

1. **Functional**: Environment runs, training produces improving rewards
2. **Correct physics**: Suction attaches/detaches box properly
3. **Reward shaping**: Agent learns approach -> pick -> place sequence within ~2000 iterations
4. **Code quality**: Follows existing codebase patterns, passes `./isaaclab.sh -f` linting

---

## Risk Factors

| Risk | Impact | Mitigation |
|------|--------|------------|
| SurfaceGripper API changes | High | Check exact data API for gripping state before implementing suction_state obs |
| CPU-only limits training speed | Medium | Use 256-512 envs, accept longer wall-clock time |
| Reward too sparse / too shaped | Medium | Start with conservative weights, tune iteratively |
| Box-goal overlap at init | Low | Add min separation check in reset event |
| UniformPoseCommandCfg may need position-only variant | Low | Wrap or extract position from 7D command (pos + quat) |

---

## Next Steps

Run `/start-work` to begin implementation following this plan.
