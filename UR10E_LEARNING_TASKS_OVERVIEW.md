# UR10 / UR10e Learning Tasks in Isaac Lab

## Scope
This summary covers UR10/UR10e manipulation tasks and whether they are wired to default RL training configs.

UR10e learning task IDs:
- `Isaac-Deploy-Reach-UR10e-v0`
- `Isaac-Deploy-GearAssembly-UR10e-2F140-v0`
- `Isaac-Deploy-GearAssembly-UR10e-2F85-v0`

UR10 learning task ID:
- `Isaac-Reach-UR10-v0`

---

## UR10 vs UR10e (short answer)
- Difference is **not only end-effector**.
- They differ in robot asset/model, joint/kinematics tuning, actuator/randomization settings, and task config families.
- End-effector tooling (suction / Robotiq 2F-85 / 2F-140) is an additional axis on top of base arm type.

## 1) Isaac-Deploy-Reach-UR10e-v0

### Observation space (policy)
- `joint_pos` (UR10e joints)
- `joint_vel` (UR10e joints)
- `pose_command` (target EE pose command from `UniformPoseCommand`, shape `(x,y,z,qw,qx,qy,qz)`)
- Effective dimensionality is typically `6 + 6 + 7 = 19` for UR10e.

### Base algorithm
- `RSL-RL` on-policy `PPO` (non-recurrent actor-critic).
- No SAC config is registered for this task.

### Reward design
- `end_effector_keypoint_tracking` (negative distance term): `-1.5`
- `end_effector_keypoint_tracking_exp` (positive exponential shaping): `+1.5`
- `action_rate_l2`: `-0.005`
- `action_l2`: `-0.005`

### Network design
- Actor MLP: `[256, 128, 64]`, ELU
- Critic MLP: `[256, 128, 64]`, ELU
- PPO: `num_steps_per_env=512`, `max_iterations=1500`, `lr=5e-4`, `gamma=0.99`, `lam=0.95`

### Reference files
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/joint_pos_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/reach_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/mdp/rewards.py`

### Commands
Train:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Deploy-Reach-UR10e-v0 --headless
```
Play (validation):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Deploy-Reach-UR10e-Play-v0 \
  --checkpoint <path/to/model_xxx.pt> --num_envs 1
```

---

## 2) Isaac-Deploy-GearAssembly-UR10e-2F140-v0 / 3) Isaac-Deploy-GearAssembly-UR10e-2F85-v0

These two tasks share the same learning structure and differ mainly in gripper model/parameters.

### Observation space
- Policy group:
  - `joint_pos` (UR10e 6 arm joints)
  - `joint_vel` (UR10e 6 arm joints)
  - `gear_shaft_pos` (3)
  - `gear_shaft_quat` (4)
  - Approx. policy dim: `6 + 6 + 3 + 4 = 19`
- Critic group (asymmetric critic):
  - Policy terms plus `gear_pos` (3), `gear_quat` (4)
  - Approx. critic dim: `26`

### Base algorithm
- `RSL-RL` on-policy `Recurrent PPO` (LSTM actor-critic).
- No SAC config is registered for these tasks.

### Reward design
- `end_effector_gear_keypoint_tracking`: `-1.5`
- `end_effector_gear_keypoint_tracking_exp`: `+1.5`
- `action_rate_l2`: `-5e-6`
- Additional reset/termination logic for dropped gear and orientation threshold is used to stabilize training.

### Network design
- Recurrent PPO actor-critic (`RslRlPpoActorCriticRecurrentCfg`)
- Actor MLP: `[256, 128, 64]`, ELU
- Critic MLP: `[256, 128, 64]`, ELU
- RNN: `LSTM`, hidden size `256`, layers `2`
- PPO: `num_steps_per_env=512`, `max_iterations=1500`, `lr=5e-4`, `gamma=0.99`, `lam=0.95`, `num_mini_batches=16`

### Reference files
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/config/ur_10e/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/gear_assembly_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/gear_assembly/config/ur_10e/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/mdp/rewards.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/mdp/terminations.py`

### Commands
Train (2F-140):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Deploy-GearAssembly-UR10e-2F140-v0 --headless
```
Play (2F-140 validation):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Deploy-GearAssembly-UR10e-2F140-Play-v0 \
  --checkpoint <path/to/model_xxx.pt> --num_envs 1
```
Train (2F-85):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Deploy-GearAssembly-UR10e-2F85-v0 --headless
```
Play (2F-85 validation):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Deploy-GearAssembly-UR10e-2F85-Play-v0 \
  --checkpoint <path/to/model_xxx.pt> --num_envs 1
```

---

## 4) Isaac-Reach-UR10-v0

### Observation space (policy)
- `joint_pos_rel`
- `joint_vel_rel`
- `pose_command` (UniformPoseCommand, 7D pose command)
- `last_action`
- Approx. policy dim is typically `6 + 6 + 7 + 6 = 25` for 6-DOF UR10.

### Base algorithm
- Multiple defaults are registered:
  - `RSL-RL PPO`
  - `RL-Games PPO`
  - `SKRL PPO`
- No SAC config is registered by default.

### Reward design
- `end_effector_position_tracking` (negative position error): `-0.2`
- `end_effector_position_tracking_fine_grained` (tanh shaping): `+0.1`
- `end_effector_orientation_tracking` (negative orientation error): `-0.1`
- `action_rate_l2`: `-1e-4` (curriculum tightens to `-0.005`)
- `joint_vel_l2`: `-1e-4` (curriculum tightens to `-0.001`)

### Network design (RSL-RL default)
- Actor MLP: `[64, 64]`, ELU
- Critic MLP: `[64, 64]`, ELU
- PPO: `num_steps_per_env=24`, `max_iterations=1000`, `lr=1e-3`, `gamma=0.99`, `lam=0.95`

### Reference files
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/joint_pos_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/agents/rl_games_ppo_cfg.yaml`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/ur_10/agents/skrl_ppo_cfg.yaml`

### Commands
Train (RSL-RL):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-UR10-v0 --headless
```
Play (validation):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-UR10-Play-v0 \
  --checkpoint <path/to/model_xxx.pt> --num_envs 1
```

---

## 5) Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0 / 6) Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0

### Important classification
- These are `ManagerBasedRLEnv` task IDs, but in current config they are **not wired to a default RL training recipe**:
  - no `rsl_rl_cfg_entry_point` / `skrl_cfg_entry_point` in registration
  - environment config sets `rewards=None`, `commands=None`, `events=None`, `curriculum=None`
- Practically, they are control/interaction IK stack environments unless you add a custom RL config and reward manager.

### Observation/action design
- Observation groups are dict-style task signals for stacking state (`policy`, `rgb_camera`, `subtask_terms`)
- IK-relative arm control through `DifferentialInverseKinematicsActionCfg`
- UR10 suction variants force CPU device in cfg (`self.device = "cpu"`), which also affects training setup.

### Reference files
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/ur10_gripper/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/ur10_gripper/stack_ik_rel_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/ur10_gripper/stack_joint_pos_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/stack_env_cfg.py`

### Commands
Play / interaction check (recommended path):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Stack-Cube-UR10-Long-Suction-IK-Rel-v0 \
  --num_envs 1
```
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0 \
  --num_envs 1
```
Training note:
- Default registration does not provide `rsl_rl_cfg_entry_point` for these two IDs, so standard RL `train.py` requires custom agent-config wiring first.

---

## Notes
- `Play` and `ROS-Inference` task IDs are evaluation/deployment paths and are excluded from the learning-task sections.
- UR10e learning tasks above are PPO-family in current repo snapshot.
- UR10 stack IK tasks need additional RL wiring if you want to use standard `train.py` RL pipelines.
