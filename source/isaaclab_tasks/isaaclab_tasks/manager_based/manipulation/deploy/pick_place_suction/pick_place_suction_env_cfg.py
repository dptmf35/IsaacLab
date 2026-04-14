# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

import isaaclab_tasks.manager_based.manipulation.deploy.pick_place_suction.mdp as mdp

##
# Scene definition
##


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the pick-and-place scene with a UR10 suction gripper arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # robots (filled in by robot-specific config)
    robot: ArticulationCfg = MISSING
    surface_gripper: SurfaceGripperCfg = MISSING

    # end-effector frame (filled in by robot-specific config)
    ee_frame: FrameTransformerCfg = MISSING

    # box to pick and place — scaled 3x so it's a reasonable 12cm cube
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.06], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(3.0, 3.0, 3.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # Flat lab table — box rests on this surface
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP.

    The goal position is sampled uniformly on the table surface.
    We reuse UniformPoseCommandCfg but fix orientation to facing down.
    """

    goal_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # filled by robot-specific config (e.g., "ee_link")
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7),
            pos_y=(-0.25, 0.25),
            pos_z=(0.06, 0.06),  # box center height on table (3x scaled block)
            roll=(3.14, 3.14),   # facing down
            pitch=(0.0, 0.0),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # robot state
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # end-effector state
        ee_pos_w = ObsTerm(
            func=mdp.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("ee_frame")},
        )
        ee_quat_w = ObsTerm(
            func=mdp.ee_quat_w,
            params={"asset_cfg": SceneEntityCfg("ee_frame")},
        )

        # box state
        box_pos_w = ObsTerm(
            func=mdp.object_pos_w,
            params={"asset_cfg": SceneEntityCfg("box")},
        )
        box_quat_w = ObsTerm(
            func=mdp.object_quat_w,
            params={"asset_cfg": SceneEntityCfg("box")},
        )

        # goal position (from command manager)
        goal_pos = ObsTerm(func=mdp.goal_pos_command, params={"command_name": "goal_pose"})

        # suction state: -1=open, 0=closing, 1=closed
        suction_state = ObsTerm(
            func=mdp.suction_state,
            params={"asset_cfg": SceneEntityCfg("surface_gripper")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.125, 0.125),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_box = EventTerm(
        func=mdp.reset_box_random,
        mode="reset",
        params={
            # x: in front of robot, y: side-to-side, z: box half-height above table (3x scale = ~0.06m half-height)
            "pose_range": {"x": (0.3, 0.7), "y": (-0.25, 0.25), "z": (0.06, 0.06)},
            "asset_cfg": SceneEntityCfg("box"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # approach: reward for moving EE toward the box
    approach_box = RewTerm(
        func=mdp.approach_box_reward,
        weight=1.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "box_cfg": SceneEntityCfg("box"),
            "surface_gripper_cfg": SceneEntityCfg("surface_gripper"),
            "alpha": 10.0,
        },
    )

    # grasp: reward for activating suction near the box
    grasp = RewTerm(
        func=mdp.grasp_reward,
        weight=0.5,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "box_cfg": SceneEntityCfg("box"),
            "surface_gripper_cfg": SceneEntityCfg("surface_gripper"),
            "grasp_threshold": 0.05,
        },
    )

    # lift: reward for lifting the box
    lift_box = RewTerm(
        func=mdp.lift_box_reward,
        weight=2.0,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "surface_gripper_cfg": SceneEntityCfg("surface_gripper"),
            "table_height": 0.06,   # box bottom = 0 (table surface), center at 0.06 (3x scaled block)
            "lift_target": 0.2,
        },
    )

    # place: reward for moving box toward goal
    place_box = RewTerm(
        func=mdp.place_box_reward,
        weight=3.0,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "surface_gripper_cfg": SceneEntityCfg("surface_gripper"),
            "command_name": "goal_pose",
            "beta": 10.0,
            "lift_threshold": 0.1,   # box center must be above 0.1m (lifted off table)
        },
    )

    # success: large bonus when box is placed at goal
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=1.0,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "command_name": "goal_pose",
            "success_threshold": 0.05,
        },
    )

    # regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    action = RewTerm(func=mdp.action_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    box_dropped = DoneTerm(
        func=mdp.box_dropped,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "min_height": -0.1,
        },
    )

    box_at_goal = DoneTerm(
        func=mdp.box_at_goal,
        params={
            "box_cfg": SceneEntityCfg("box"),
            "command_name": "goal_pose",
            "success_threshold": 0.05,
        },
    )


##
# Environment configuration
##


@configclass
class PickPlaceSuctionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the UR10 suction gripper pick-and-place environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=256, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        # CPU required for suction gripper
        self.device = "cpu"
