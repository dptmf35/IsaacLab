# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the pick-and-place suction environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def box_dropped(env: ManagerBasedRLEnv, box_cfg: SceneEntityCfg, min_height: float = -0.1) -> torch.Tensor:
    """Terminate if the box falls below the table.

    Args:
        env: The environment instance.
        box_cfg: Configuration for the box RigidObject.
        min_height: Minimum allowed height in world frame. Defaults to -0.1.

    Returns:
        Boolean tensor of shape (num_envs,). True means terminate.
    """
    box: RigidObject = env.scene[box_cfg.name]
    return box.data.root_pos_w[:, 2] < min_height


def box_at_goal(
    env: ManagerBasedRLEnv,
    box_cfg: SceneEntityCfg,
    command_name: str,
    success_threshold: float = 0.05,
) -> torch.Tensor:
    """Terminate (success) when box is placed within success_threshold of goal in XY plane.

    Args:
        env: The environment instance.
        box_cfg: Configuration for the box RigidObject.
        command_name: Name of the goal pose command.
        success_threshold: Maximum XY distance to goal for success. Defaults to 0.05.

    Returns:
        Boolean tensor of shape (num_envs,). True means success.
    """
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_xy = box.data.root_pos_w[:, :2]

    from isaaclab.assets import Articulation

    command = env.command_manager.get_command(command_name)  # (num_envs, 7): in robot base frame
    robot: Articulation = env.scene["robot"]
    goal_pos_xy = robot.data.root_pos_w[:, :2] + command[:, :2]

    dist_xy = torch.norm(box_pos_xy - goal_pos_xy, dim=-1)
    return dist_xy < success_threshold
