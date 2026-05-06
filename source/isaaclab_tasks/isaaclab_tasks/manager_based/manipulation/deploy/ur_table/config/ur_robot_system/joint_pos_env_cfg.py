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
            debug_vis=True,
            visualizer_cfg=_MARKER_CFG,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/UR_Robot_System/ur3_with_gripper/short_gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.05, 0.0, 0.0],
                        # short_gripper의 x축이 tool 방향(world down)이므로
                        # R_y(+90°)를 적용해 z축이 tool 방향을 가리키도록 정렬
                        # → 표준 z-down 컨벤션으로 통일 (0.5, 0.5, 0.5, 0.5), 
                        rot=(0.7071, 0.0, 0.7071, 0.0) # wxyz: z=down, x=우 (goal frame과 정렬)
                    ),
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
        self.actions.table_action = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint"],
            scale=0.1,
            use_zero_offset=False,
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
        self.episode_length_s = 4.0
        self.commands.goal_pose.resampling_time_range = (4.0, 4.0)

        # ── Observations: goal_quat 추가 (yaw 변동 → orientation 학습 필요) ──
        self.observations.policy.goal_quat = ObsTerm(
            func=mdp.goal_quat_command,
            params={"command_name": "goal_pose"},
        )

        # ── Replace rewards: position align + regularisation ─────────────────
        self.rewards.action_rate.weight = -0.01
        self.rewards.action.weight = -0.01

        # [1] Coarse approach: linear -dist → 거리에 무관하게 일정한 gradient
        #     pos_x 범위가 ±0.9m로 넓어서 exp 단독으론 먼 거리 gradient가 0에 가까움
        #     → 이 term이 멀 때 (dist > ~0.3m) 방향 신호 담당
        self.rewards.ee_pos_approach = RewTerm(
            func=mdp.ee_neg_dist,
            weight=0.3,
            params={
                "command_name": "goal_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        # [2] Precision bonus: exp(-10*dist) → 0.3m 이내 진입 시 sharp 보상
        #     alpha=10은 "정밀 구간 전용" — linear term이 먼 거리 커버하므로 높아도 OK
        self.rewards.ee_pos_precision = RewTerm(
            func=mdp.ee_goal_pos_reward,
            weight=1.0,
            params={
                "command_name": "goal_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "alpha": 10.0,
            },
        )

        # [3] Orientation reward (yaw ±0.5 rad 변동 → 방향 정렬 학습)
        self.rewards.ee_ori_align = RewTerm(
            func=mdp.ee_goal_ori_reward,
            weight=0.3,
            params={
                "command_name": "goal_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        # ── Success termination ───────────────────────────────────────────────
        self.terminations.ee_success = DoneTerm(
            func=mdp.ee_at_goal_hold,
            params={
                "command_name": "goal_pose",
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "threshold": 0.03,   # 3 cm
                "hold_steps": 5,     # ~0.17s 연속 유지 필요 (decimation=4, dt=1/120)
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
