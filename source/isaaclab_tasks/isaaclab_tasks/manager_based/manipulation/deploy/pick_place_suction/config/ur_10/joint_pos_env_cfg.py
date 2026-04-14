# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 Long Suction robot configuration for the pick-and-place environment."""

from isaaclab.assets import SurfaceGripperCfg
from isaaclab.envs.mdp.actions.actions_cfg import SurfaceGripperBinaryActionCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.pick_place_suction.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.deploy.pick_place_suction.pick_place_suction_env_cfg import (
    PickPlaceSuctionEnvCfg,
)

from isaaclab_assets.robots.universal_robots import UR10_LONG_SUCTION_CFG  # isort: skip


##
# Pre-defined configs
##
_MARKER_CFG = FRAME_MARKER_CFG.copy()
_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"


@configclass
class UR10LongSuctionPickPlaceEnvCfg(PickPlaceSuctionEnvCfg):
    """UR10 Long Suction robot configuration for pick-and-place.

    Uses UR10_LONG_SUCTION_CFG which loads the UR10 USD with the Long_Suction gripper variant.
    SurfaceGripper prim is at {ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper.
    End-effector offset: 0.22m along local x-axis from ee_link.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # CPU required for surface gripper
        self.device = "cpu"

        # Set robot with long suction gripper variant.
        # Override init joint pos so EE faces downward toward the table.
        # UR10_LONG_SUCTION_CFG default has wrist_2_joint=+1.5707 which makes EE face up.
        # Setting wrist_2_joint=-1.5707 (same as stack task default) makes it face down.
        _robot_cfg = UR10_LONG_SUCTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _robot_cfg.init_state.joint_pos = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5707,
            "elbow_joint": 1.5707,
            "wrist_1_joint": -1.5707,
            "wrist_2_joint": -1.5707,  # -90 deg: EE faces down (not up)
            "wrist_3_joint": 0.0,
        }
        self.scene.robot = _robot_cfg

        # Configure surface gripper scene entity
        self.scene.surface_gripper = SurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ee_link/SurfaceGripper",
            max_grip_distance=0.0075,
            shear_force_limit=5000.0,
            coaxial_force_limit=5000.0,
            retry_interval=0.05,
        )

        # End-effector frame: track ee_link with 0.22m offset along x (tip of long suction cup)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=_MARKER_CFG,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.22, 0.0, 0.0]),
                ),
            ],
        )

        # Arm action: incremental joint position control
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.0625, use_zero_offset=True
        )

        # Gripper action: binary suction on/off
        self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

        # Set command body name for goal pose
        self.commands.goal_pose.body_name = "ee_link"

        # Smaller scene for CPU simulation
        self.scene.num_envs = 256
        self.scene.env_spacing = 3.0


@configclass
class UR10LongSuctionPickPlaceEnvCfg_PLAY(UR10LongSuctionPickPlaceEnvCfg):
    """Play configuration: fewer environments, no randomization noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
