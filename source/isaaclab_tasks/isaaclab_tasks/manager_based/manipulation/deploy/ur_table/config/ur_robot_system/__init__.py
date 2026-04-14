# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym environment registration for UR Robot System pick-and-place."""

import gymnasium as gym

from . import agents
from .joint_pos_env_cfg import (
    URRobotSystemEnvCfg,
    URRobotSystemEnvCfg_PLAY,
    URTableAlignEnvCfg,
    URTableAlignEnvCfg_PLAY,
)

gym.register(
    id="Isaac-URTable-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:URRobotSystemEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:URTablePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-URTable-PickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:URRobotSystemEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:URTablePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-URTable-Align-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:URTableAlignEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:URTablePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-URTable-Align-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:URTableAlignEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:URTablePPORunnerCfg",
    },
)
