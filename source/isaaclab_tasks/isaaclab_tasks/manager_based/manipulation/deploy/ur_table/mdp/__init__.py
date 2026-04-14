# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the UR table pick-and-place environment."""

# Re-use all standard deploy MDP terms (joint_pos, joint_vel, action_rate_l2, etc.)
from isaaclab_tasks.manager_based.manipulation.deploy.mdp import *  # noqa: F401, F403

# Re-use pick-and-place suction MDP terms (rewards, events, observations, terminations)
from isaaclab_tasks.manager_based.manipulation.deploy.pick_place_suction.mdp import *  # noqa: F401, F403

# UR-table-specific MDP terms (align task)
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
