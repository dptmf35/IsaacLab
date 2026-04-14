# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the pick-and-place suction environment."""

# Re-export standard MDP terms from deploy mdp
from isaaclab_tasks.manager_based.manipulation.deploy.mdp import *  # noqa: F401, F403

# Re-export custom terms for this task
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
