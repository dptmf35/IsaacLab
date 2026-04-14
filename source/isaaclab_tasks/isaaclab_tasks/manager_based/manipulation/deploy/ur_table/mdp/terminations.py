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
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins  # env-local
    goal_pos = env.command_manager.get_command(command_name)[:, :3]  # env-local
    dist = torch.norm(ee_pos - goal_pos, dim=-1)
    return dist < threshold
