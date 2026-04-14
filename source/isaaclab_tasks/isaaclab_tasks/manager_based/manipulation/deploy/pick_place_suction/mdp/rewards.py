# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the pick-and-place suction environment.

Reward structure (phased dense rewards):
  1. approach_box: Move EE toward box (active when suction OFF)
  2. grasp:        Small bonus when suction activates near box
  3. lift_box:     Reward for lifting box above table (active when suction ON + grasped)
  4. place_box:    Move box toward goal position (active when box is lifted)
  5. success_bonus: Large sparse reward for successful placement
  6. action/action_rate: Regularization penalties
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, SurfaceGripper
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """Helper: Get end-effector position in world frame. Returns (num_envs, 3)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :]


def _get_box_pos(env: ManagerBasedRLEnv, box_cfg: SceneEntityCfg) -> torch.Tensor:
    """Helper: Get box position in world frame. Returns (num_envs, 3)."""
    box: RigidObject = env.scene[box_cfg.name]
    return box.data.root_pos_w


def _get_gripper_state(env: ManagerBasedRLEnv, surface_gripper_cfg: SceneEntityCfg) -> torch.Tensor:
    """Helper: Get suction state (-1=open, 0=closing, 1=closed). Returns (num_envs,)."""
    gripper: SurfaceGripper = env.scene[surface_gripper_cfg.name]
    return gripper.state.float()


def _get_goal_pos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Helper: Get goal position in world frame. Returns (num_envs, 3).

    UniformPoseCommand returns commands in the robot base frame.
    We convert to world frame using the robot's root position.
    """
    from isaaclab.assets import Articulation

    command = env.command_manager.get_command(command_name)  # (num_envs, 7): [x, y, z, qw, qx, qy, qz]
    robot: Articulation = env.scene["robot"]
    # command[:, :3] is in robot base frame; robot root is approx at env origin (no yaw rotation)
    goal_pos_w = robot.data.root_pos_w + command[:, :3]
    return goal_pos_w


class approach_box_reward(ManagerTermBase):
    """Exponential dense reward for moving EE toward the box.

    Only active when the suction gripper is OFF (state == -1).
    Encourages the agent to approach the box before activating suction.

    reward = exp(-alpha * ||ee_pos - box_pos||)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ee_frame_cfg: SceneEntityCfg = cfg.params["ee_frame_cfg"]
        self.box_cfg: SceneEntityCfg = cfg.params["box_cfg"]
        self.surface_gripper_cfg: SceneEntityCfg = cfg.params["surface_gripper_cfg"]
        self.alpha: float = cfg.params.get("alpha", 10.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ee_frame_cfg: SceneEntityCfg,
        box_cfg: SceneEntityCfg,
        surface_gripper_cfg: SceneEntityCfg,
        alpha: float,
    ) -> torch.Tensor:
        ee_pos = _get_ee_pos(env, ee_frame_cfg)
        box_pos = _get_box_pos(env, box_cfg)
        gripper_state = _get_gripper_state(env, surface_gripper_cfg)

        dist = torch.norm(ee_pos - box_pos, dim=-1)
        reward = torch.exp(-alpha * dist)

        # Only active when suction is OFF (state == -1)
        active = (gripper_state == -1.0).float()
        return reward * active


class grasp_reward(ManagerTermBase):
    """Small reward when suction activates close to the box.

    Encourages the agent to activate suction only when near the box.

    reward = 1.0 if (suction ON) and (dist(ee, box) < grasp_threshold)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ee_frame_cfg: SceneEntityCfg = cfg.params["ee_frame_cfg"]
        self.box_cfg: SceneEntityCfg = cfg.params["box_cfg"]
        self.surface_gripper_cfg: SceneEntityCfg = cfg.params["surface_gripper_cfg"]
        self.grasp_threshold: float = cfg.params.get("grasp_threshold", 0.05)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ee_frame_cfg: SceneEntityCfg,
        box_cfg: SceneEntityCfg,
        surface_gripper_cfg: SceneEntityCfg,
        grasp_threshold: float,
    ) -> torch.Tensor:
        ee_pos = _get_ee_pos(env, ee_frame_cfg)
        box_pos = _get_box_pos(env, box_cfg)
        gripper_state = _get_gripper_state(env, surface_gripper_cfg)

        dist = torch.norm(ee_pos - box_pos, dim=-1)

        # Reward when suction is ON (closing or closed) and near the box
        suction_on = (gripper_state >= 0.0).float()
        near_box = (dist < grasp_threshold).float()
        return suction_on * near_box


class lift_box_reward(ManagerTermBase):
    """Reward for lifting the box above the table.

    Active only when suction is ON (gripper state >= 0) and box is above table.

    reward = clip(box_z - table_height, 0, lift_target) / lift_target
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.box_cfg: SceneEntityCfg = cfg.params["box_cfg"]
        self.surface_gripper_cfg: SceneEntityCfg = cfg.params["surface_gripper_cfg"]
        self.table_height: float = cfg.params.get("table_height", 0.0203)
        self.lift_target: float = cfg.params.get("lift_target", 0.15)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        box_cfg: SceneEntityCfg,
        surface_gripper_cfg: SceneEntityCfg,
        table_height: float,
        lift_target: float,
    ) -> torch.Tensor:
        box_pos = _get_box_pos(env, box_cfg)
        gripper_state = _get_gripper_state(env, surface_gripper_cfg)

        # box_z in world frame (origins[2] is the env floor offset, table is at z=table_height relative to env)
        box_z = box_pos[:, 2]

        lift_height = torch.clamp(box_z - table_height, min=0.0, max=lift_target) / lift_target

        # Active only when suction is ON
        suction_on = (gripper_state >= 0.0).float()
        return lift_height * suction_on


class place_box_reward(ManagerTermBase):
    """Exponential reward for moving the grasped box toward the goal position.

    Active only when suction is ON and box is lifted above a threshold.

    reward = exp(-beta * ||box_pos - goal_pos||)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.box_cfg: SceneEntityCfg = cfg.params["box_cfg"]
        self.surface_gripper_cfg: SceneEntityCfg = cfg.params["surface_gripper_cfg"]
        self.command_name: str = cfg.params["command_name"]
        self.beta: float = cfg.params.get("beta", 10.0)
        self.lift_threshold: float = cfg.params.get("lift_threshold", 0.05)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        box_cfg: SceneEntityCfg,
        surface_gripper_cfg: SceneEntityCfg,
        command_name: str,
        beta: float,
        lift_threshold: float,
    ) -> torch.Tensor:
        box_pos = _get_box_pos(env, box_cfg)
        goal_pos = _get_goal_pos(env, command_name)
        gripper_state = _get_gripper_state(env, surface_gripper_cfg)

        # XY distance to goal
        dist_xy = torch.norm(box_pos[:, :2] - goal_pos[:, :2], dim=-1)
        reward = torch.exp(-beta * dist_xy)

        # Active when suction is ON and box is lifted
        suction_on = (gripper_state >= 0.0).float()
        box_lifted = (box_pos[:, 2] > lift_threshold).float()
        return reward * suction_on * box_lifted


class success_bonus(ManagerTermBase):
    """Large sparse reward when box is successfully placed at the goal.

    Condition: ||box_pos_xy - goal_pos_xy|| < success_threshold

    reward = 10.0 if success, else 0.0
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.box_cfg: SceneEntityCfg = cfg.params["box_cfg"]
        self.command_name: str = cfg.params["command_name"]
        self.success_threshold: float = cfg.params.get("success_threshold", 0.05)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        box_cfg: SceneEntityCfg,
        command_name: str,
        success_threshold: float,
    ) -> torch.Tensor:
        box_pos = _get_box_pos(env, box_cfg)
        goal_pos = _get_goal_pos(env, command_name)

        dist_xy = torch.norm(box_pos[:, :2] - goal_pos[:, :2], dim=-1)
        success = (dist_xy < success_threshold).float()
        return success * 10.0
