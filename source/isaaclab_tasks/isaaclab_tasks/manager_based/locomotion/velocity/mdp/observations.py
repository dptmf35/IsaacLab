# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for locomotion velocity environments."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, gait_period: int = 20) -> torch.Tensor:
    """Encode gait phase as [sin, cos] using the environment step counter.

    The phase cycles from 0 to 2π over ``gait_period`` control steps, providing
    the BC policy with an explicit periodic signal so it can learn gait timing
    without having to infer it from hidden state alone.

    Args:
        env: The environment instance. Uses ``env.episode_length_buf`` (per-env
             step counter that resets on episode reset) to track phase.
        gait_period: Number of control steps per full gait cycle.
                     Default 20 ≈ 0.4 s at 50 Hz control (A1 trot).

    Returns:
        Tensor of shape ``(num_envs, 2)`` containing ``[sin(phase), cos(phase)]``.
    """
    phase = 2.0 * math.pi * (env.episode_length_buf % gait_period).float() / gait_period
    return torch.stack([phase.sin(), phase.cos()], dim=-1)
