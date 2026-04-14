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
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins  # env-local
    goal_pos = env.command_manager.get_command(command_name)[:, :3]  # env-local
    dist = torch.norm(ee_pos - goal_pos, dim=-1)
    return torch.exp(-alpha * dist)


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
