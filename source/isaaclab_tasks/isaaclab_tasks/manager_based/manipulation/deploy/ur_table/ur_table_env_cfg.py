# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base environment configuration for the UR robot system with suction gripper and X-axis prismatic table.

The full system (UR arm + suction gripper + prismatic table) is loaded as a single
ArticulationCfg from the local Digital Twin USD file.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, SurfaceGripperCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.deploy.ur_table.mdp as mdp

##
# Scene definition
##


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene with the UR robot system (arm + suction gripper + prismatic table)."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # The full robot system as a single articulation (filled by robot-specific cfg)
    robot: ArticulationCfg = MISSING

    # SurfaceGripper entity (prim path filled by robot-specific cfg)
    surface_gripper: SurfaceGripperCfg = MISSING

    # End-effector frame sensor (filled by robot-specific cfg)
    ee_frame: FrameTransformerCfg = MISSING

    # Object to pick and place (disabled until table surface height is confirmed)
    # box = RigidObjectCfg(...)


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Goal pose command sampled uniformly on the table surface."""

    goal_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # filled by robot-specific cfg (e.g., "ee_link")
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # 보수적 reachable workspace: 원점에서 최대 ~0.45m 이내 직사각형
            # 코너 거리 = sqrt(0.4² + 0.2²) ≈ 0.45m → UR3 reach 안쪽
            # TODO: random_agent 실행 후 실제 EE 도달 범위 확인해서 조정
            pos_x=(-0.9, 0.9),
            pos_y=(-0.4, -0.3),
            pos_z=(0.5, 0.5),
            roll=(3.14, 3.14),
            pitch=(0.0, 0.0),
            yaw=(-0.5, 0.5),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # UR arm joint position control (relative mode)
    arm_action: ActionTerm = MISSING

    # X-axis prismatic table control
    table_action: ActionTerm = MISSING

    # Suction gripper on/off
    gripper_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Arm + table joint states
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # End-effector pose in world frame
        ee_pos_w = ObsTerm(
            func=mdp.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("ee_frame")},
        )
        ee_quat_w = ObsTerm(
            func=mdp.ee_quat_w,
            params={"asset_cfg": SceneEntityCfg("ee_frame")},
        )

        # Goal position command
        goal_pos = ObsTerm(func=mdp.goal_pos_command, params={"command_name": "goal_pose"})

        # Suction gripper state (-1=open, 0=closing, 1=closed)
        suction_state = ObsTerm(
            func=mdp.suction_state,
            params={"asset_cfg": SceneEntityCfg("surface_gripper")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Randomization and reset events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    # reset_box disabled until box is re-added to scene


@configclass
class RewardsCfg:
    """Reward terms for pick-and-place with suction gripper."""

    # Regularization only (box rewards disabled until box is re-added)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    action = RewTerm(func=mdp.action_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class URTableEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for UR robot system pick-and-place environment."""

    scene: SceneCfg = SceneCfg(num_envs=256, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 4.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 120.0
        # SurfaceGripper requires CPU pipeline
        self.device = "cpu"
