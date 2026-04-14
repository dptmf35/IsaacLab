# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym environment registration for UR10 Suction Pick-and-Place."""

import gymnasium as gym

from . import agents
from .joint_pos_env_cfg import UR10LongSuctionPickPlaceEnvCfg, UR10LongSuctionPickPlaceEnvCfg_PLAY

gym.register(
    id="Isaac-Deploy-PickPlace-UR10-Suction-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10LongSuctionPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Deploy-PickPlace-UR10-Suction-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10LongSuctionPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
)
