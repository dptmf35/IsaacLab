# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR Robot System configuration for the pick-and-place environment.

This config wires the custom Digital Twin USD (ur_gripper.usd) into the
URTableEnvCfg base. The USD contains:
  - UR arm (6-DOF)
  - Suction gripper (SurfaceGripper prim)
  - X-axis prismatic table joint

All share a single ArticulationRoot under "UR_Robot_System".

TODO items (verify in Isaac Sim Stage panel before running):
  1. ``table_x_joint``      → actual prismatic joint name in the USD
  2. SurfaceGripper prim path → actual path inside the USD hierarchy
  3. ee_link prim path         → actual end-effector link name
  4. FrameTransformerCfg base prim → actual base/root link name
"""

from isaaclab.assets import SurfaceGripperCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
    SurfaceGripperBinaryActionCfg,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.ur_table.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.deploy.ur_table.ur_table_env_cfg import URTableEnvCfg

from isaaclab_assets.robots.universal_robots import UR_ROBOT_SYSTEM_CFG  # isort: skip


_MARKER_CFG = FRAME_MARKER_CFG.copy()
_MARKER_CFG.markers["frame"].scale = (0.1, 0.1, 0.1)
_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"


@configclass
class URRobotSystemEnvCfg(URTableEnvCfg):
    """Concrete env config for the custom UR robot system USD."""

    def __post_init__(self):
        super().__post_init__()

        # ── Robot articulation ────────────────────────────────────────────────
        # The USD root prim "UR_Robot_System" becomes {ENV_REGEX_NS}/UR_Robot_System
        self.scene.robot = UR_ROBOT_SYSTEM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/UR_Robot_System"
        )

        # ── SurfaceGripper ───────────────────────────────────────────────────
        # TODO: update prim_path to match the SurfaceGripper location in your USD.
        #       Common pattern: "{ENV_REGEX_NS}/UR_Robot_System/ee_link/SurfaceGripper"
        #       Open Isaac Sim → Stage panel → find the prim with type "SurfaceGripper"
        self.scene.surface_gripper = SurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/UR_Robot_System/ur3_with_gripper/short_gripper",
            max_grip_distance=0.0075,
            shear_force_limit=5000.0,
            coaxial_force_limit=5000.0,
            retry_interval=0.05,
        )

        # ── End-effector frame sensor ─────────────────────────────────────────
        # TODO: update prim_path (base link) and target ee_link name to match your USD.
        #       "base_link" is the fixed root body; "ee_link" is the tool-center-point link.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/UR_Robot_System/ur3_with_gripper/base_link",
            debug_vis=False,
            visualizer_cfg=_MARKER_CFG,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/UR_Robot_System/ur3_with_gripper/short_gripper",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.15, 0.0, 0.0]),
                ),
            ],
        )

        # ── Actions ──────────────────────────────────────────────────────────
        # UR arm: 6 joints, incremental joint position control
        self.actions.arm_action = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint",
                         "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            scale=0.0625,
            use_zero_offset=True,
        )

        # X-axis prismatic table: absolute position control
        # TODO: replace "table_x_joint" with the actual joint name from your USD
        self.actions.table_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint"],
            scale=1.0,
        )

        # Suction gripper: binary on/off
        self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

        # ── Command goal body ─────────────────────────────────────────────────
        # TODO: update to the actual end-effector link name if different
        self.commands.goal_pose.body_name = "short_gripper"


@configclass
class URRobotSystemEnvCfg_PLAY(URRobotSystemEnvCfg):
    """Play/evaluation config: fewer envs, no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False


@configclass
class URTableAlignEnvCfg(URRobotSystemEnvCfg):
    """Lightweight EE-align task: move the end-effector to a sampled goal pose.

    No object/box involved — purely teach the arm to reach any goal position
    within the table workspace.  Gripper action is kept in the action space
    but the reward signal ignores it, so the policy will learn to ignore it.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── Shorter episode — simpler task than full pick-place ───────────────
        self.episode_length_s = 8.0
        self.commands.goal_pose.resampling_time_range = (6.0, 6.0)

        # ── Goal quat added to observations ──────────────────────────────────
        self.observations.policy.goal_quat = ObsTerm(
            func=mdp.goal_quat_command,
            params={"command_name": "goal_pose"},
        )

        # ── Replace rewards: position align + regularisation ─────────────────
        # Remove pick-place-only reg terms (inherit action_rate, action)
        self.rewards.action_rate.weight = -0.01
        self.rewards.action.weight = -0.01

        # Main task reward: EE position → goal position
        self.rewards.ee_pos_align = RewTerm(
            func=mdp.ee_goal_pos_reward,
            weight=1.0,
            params={
                "command_name": "goal_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "alpha": 10.0,
            },
        )

        # Optional orientation reward (low weight to not overwhelm position)
        self.rewards.ee_ori_align = RewTerm(
            func=mdp.ee_goal_ori_reward,
            weight=0.2,
            params={
                "command_name": "goal_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        # ── Success termination ───────────────────────────────────────────────
        self.terminations.ee_success = DoneTerm(
            func=mdp.ee_at_goal,
            params={
                "command_name": "goal_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "threshold": 0.03,  # 3 cm
            },
        )


@configclass
class URTableAlignEnvCfg_PLAY(URTableAlignEnvCfg):
    """Play/evaluation config for the align task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
