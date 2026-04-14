# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the pick-and-place suction environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_box_random(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize box position on the table surface at episode reset.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        pose_range: Dictionary with keys 'x', 'y', 'z' as (min, max) ranges
                    in env-local coordinates.
        asset_cfg: Configuration for the box RigidObject.
    """
    box: RigidObject = env.scene[asset_cfg.name]

    num_envs_to_reset = len(env_ids)

    # Sample random positions
    x_range = pose_range.get("x", (0.4, 0.6))
    y_range = pose_range.get("y", (-0.2, 0.2))
    z_range = pose_range.get("z", (0.0203, 0.0203))

    x = torch.zeros(num_envs_to_reset, device=env.device).uniform_(*x_range)
    y = torch.zeros(num_envs_to_reset, device=env.device).uniform_(*y_range)
    z = torch.zeros(num_envs_to_reset, device=env.device).uniform_(*z_range)

    # Convert from env-local to world frame
    pos_w = env.scene.env_origins[env_ids].clone()
    pos_w[:, 0] += x
    pos_w[:, 1] += y
    pos_w[:, 2] += z

    # Default orientation (no rotation)
    quat_w = torch.zeros(num_envs_to_reset, 4, device=env.device)
    quat_w[:, 0] = 1.0  # w=1, x=y=z=0

    # Zero velocities
    vel_w = torch.zeros(num_envs_to_reset, 6, device=env.device)

    # Write pose + velocity to simulation (shape: (num_envs_to_reset, 13))
    root_state = torch.cat([pos_w, quat_w, vel_w], dim=-1)
    box.write_root_state_to_sim(root_state, env_ids=env_ids)
