# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the UR table align task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_goal_pos_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg,
    alpha: float = 10.0,
) -> torch.Tensor:
    """Exponential reward for moving the EE toward the goal position.

    Both the EE position and the command goal are expressed in the env-local frame
    (world position minus env origin), so no additional frame conversion is needed.

    reward = exp(-alpha * ||ee_pos - goal_pos||)

    Args:
        env: The environment instance.
        command_name: Name of the UniformPoseCommand term (returns [x,y,z,qw,qx,qy,qz]).
        ee_frame_cfg: SceneEntityCfg for the FrameTransformer sensor tracking the EE.
        alpha: Decay rate. Larger → tighter peak near goal.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import combine_frame_transforms

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # world frame

    # command는 robot root frame 기준 → 회전 포함해서 world frame으로 변환
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command[:, :3],
        command[:, 3:7],
    )

    dist = torch.norm(ee_pos_w - goal_pos_w, dim=-1)
    return torch.exp(-alpha * dist)


def ee_neg_dist(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Linear (negative distance) reward for approaching the goal.

    Returns -||ee_pos - goal_pos||, which provides a constant gradient signal
    regardless of distance.  Pair with ee_goal_pos_reward (exponential) so that:
      * Far from goal  → linear term dominates → constant pull toward goal
      * Near goal      → exponential term dominates → sharp precision bonus

    Args:
        env: The environment instance.
        command_name: Name of the UniformPoseCommand term.
        ee_frame_cfg: SceneEntityCfg for the FrameTransformer sensor tracking the EE.

    Returns:
        Reward tensor of shape (num_envs,).  Values are ≤ 0.
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import combine_frame_transforms

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command[:, :3],
        command[:, 3:7],
    )

    dist = torch.norm(ee_pos_w - goal_pos_w, dim=-1)
    return -dist


def ee_goal_ori_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for aligning the EE orientation with the goal orientation.

    Uses the absolute quaternion dot product: |q_ee · q_goal| ∈ [0, 1].
    Value is 1 when perfectly aligned, 0 when 180° apart.

    Args:
        env: The environment instance.
        command_name: Name of the UniformPoseCommand term.
        ee_frame_cfg: SceneEntityCfg for the FrameTransformer sensor tracking the EE.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]  # (num_envs, 4) wxyz, world frame
    goal_quat = env.command_manager.get_command(command_name)[:, 3:7]  # wxyz
    # Absolute dot product handles the q == -q symmetry
    dot = torch.abs((ee_quat * goal_quat).sum(dim=-1))
    return dot.clamp(0.0, 1.0)
