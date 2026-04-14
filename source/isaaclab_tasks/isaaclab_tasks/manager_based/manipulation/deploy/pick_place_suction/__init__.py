# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 Suction Pick-and-Place RL task.

This module provides a manager-based RL environment for pick-and-place with:
- UR10 robot with Long Suction gripper
- Binary suction action (on/off)
- Randomized box and goal positions on table
- Phased dense rewards (approach, grasp, lift, place, success)
"""

from .config import *  # noqa: F401, F403
