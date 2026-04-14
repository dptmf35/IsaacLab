# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the pick-and-place suction environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, SurfaceGripper
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position in env-local frame (world pos minus env origin).

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the FrameTransformer asset.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    ee_frame: FrameTransformer = env.scene[asset_cfg.name]
    pos_w = ee_frame.data.target_pos_w[:, 0, :]  # (num_envs, 3)
    return pos_w - env.scene.env_origins


def ee_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector orientation (quaternion wxyz) in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the FrameTransformer asset.

    Returns:
        Tensor of shape (num_envs, 4).
    """
    ee_frame: FrameTransformer = env.scene[asset_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]  # (num_envs, 4)


def object_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rigid object position in env-local frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the RigidObject asset.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    return obj.data.root_pos_w - env.scene.env_origins


def object_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rigid object orientation (quaternion wxyz) in world frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the RigidObject asset.

    Returns:
        Tensor of shape (num_envs, 4).
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    return obj.data.root_quat_w


def goal_pos_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Goal position from the command manager in env-local frame.

    UniformPoseCommandCfg generates commands in robot base frame (= env-local frame since robot base
    is at env origin with no rotation). Returns position part only.

    Args:
        env: The environment instance.
        command_name: Name of the command term.

    Returns:
        Tensor of shape (num_envs, 3).
    """
    command = env.command_manager.get_command(command_name)  # (num_envs, 7)
    return command[:, :3]  # already in robot-base / env-local frame


def goal_quat_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Goal orientation (quaternion wxyz) from the command manager.

    Args:
        env: The environment instance.
        command_name: Name of the command term.

    Returns:
        Tensor of shape (num_envs, 4).
    """
    command = env.command_manager.get_command(command_name)  # (num_envs, 7)
    return command[:, 3:7]  # wxyz quaternion


def suction_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Current suction gripper state as a float.

    State encoding:
        -1.0 → Open
         0.0 → Closing
         1.0 → Closed (gripping)

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the SurfaceGripper asset.

    Returns:
        Tensor of shape (num_envs, 1).
    """
    gripper: SurfaceGripper = env.scene[asset_cfg.name]
    return gripper.state.float().unsqueeze(-1)
