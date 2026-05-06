# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the UR table align task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_at_goal_hold(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg,
    threshold: float = 0.03,
    hold_steps: int = 5,
) -> torch.Tensor:
    """Terminate (success) when the EE stays within threshold for hold_steps consecutive steps.

    Prevents false positives from the slider passing through the goal position transiently.
    The EE must remain within threshold for hold_steps consecutive policy steps.

    Args:
        env: The environment instance.
        command_name: Name of the UniformPoseCommand term.
        ee_frame_cfg: SceneEntityCfg for the FrameTransformer sensor tracking the EE.
        threshold: Distance threshold in metres. Defaults to 3 cm.
        hold_steps: Number of consecutive steps within threshold required. Defaults to 5.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import combine_frame_transforms

    # Initialize persistent counter (reused across steps)
    if not hasattr(env, "_ee_goal_hold_counter"):
        env._ee_goal_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # Reset counter for envs that just started a new episode
    just_reset = env.episode_length_buf <= 1
    env._ee_goal_hold_counter[just_reset] = 0

    # Compute distance to goal
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
    at_goal = dist < threshold

    # Increment counter when at goal, reset to 0 when not
    env._ee_goal_hold_counter = torch.where(
        at_goal,
        env._ee_goal_hold_counter + 1,
        torch.zeros_like(env._ee_goal_hold_counter),
    )

    return env._ee_goal_hold_counter >= hold_steps


def ee_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Terminate (success) when the EE reaches the goal position within threshold.

    Args:
        env: The environment instance.
        command_name: Name of the UniformPoseCommand term.
        ee_frame_cfg: SceneEntityCfg for the FrameTransformer sensor tracking the EE.
        threshold: Distance threshold in metres for success. Defaults to 3 cm.

    Returns:
        Boolean tensor of shape (num_envs,).
    """
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import combine_frame_transforms

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]  # world frame

    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command[:, :3],
        command[:, 3:7],
    )

    dist = torch.norm(ee_pos_w - goal_pos_w, dim=-1)
    return dist < threshold
