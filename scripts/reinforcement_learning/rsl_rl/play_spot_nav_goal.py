# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play Spot navigation with a fixed world-frame goal pose."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play Spot navigation with a fixed goal command.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Warehouse-Spot-Play-v0", help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--pos-thresh", type=float, default=0.25, help="Arrival threshold for XY position error (meters).")
parser.add_argument("--yaw-thresh", type=float, default=0.25, help="Arrival threshold for yaw error (radians).")

# append RSL-RL cli arguments (includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _prompt_goal_from_stdin() -> tuple[float, float, float]:
    """Read goal pose from terminal input as `x y yaw`."""
    fallback = (4.5, -2.0, 1.57)
    if not sys.stdin.isatty():
        print(
            "[WARN] Non-interactive stdin detected. "
            f"Using fallback goal x={fallback[0]}, y={fallback[1]}, yaw={fallback[2]}."
        )
        return fallback

    print("[INPUT] Enter goal in world frame: x y yaw")
    print("[INPUT] Example: 4.5 -2.0 1.57")
    while True:
        raw = input("goal> ").strip()
        parts = raw.split()
        if len(parts) != 3:
            print("[WARN] Please enter exactly 3 numbers: x y yaw")
            continue
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            print("[WARN] Invalid input. Example: 4.5 -2.0 1.57")


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _set_fixed_goal(env: RslRlVecEnvWrapper, goal_x: float, goal_y: float, goal_yaw: float) -> None:
    """Overwrite the pose command term with a fixed world-frame goal."""
    term = env.unwrapped.command_manager.get_term("pose_command")
    if not hasattr(term, "pos_command_w") or not hasattr(term, "heading_command_w"):
        raise TypeError("pose_command term does not expose world-frame buffers required for fixed goal injection.")

    term.pos_command_w[:, 0] = goal_x
    term.pos_command_w[:, 1] = goal_y
    term.pos_command_w[:, 2] = term.robot.data.default_root_state[:, 2]
    term.heading_command_w[:] = goal_yaw
    # Keep command from being randomly resampled by the term's internal timer.
    term.time_left[:] = 1.0e9


def _compute_goal_error(env: RslRlVecEnvWrapper, goal_x: float, goal_y: float, goal_yaw: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env position/yaw error in world frame."""
    term = env.unwrapped.command_manager.get_term("pose_command")
    root_xy = term.robot.data.root_pos_w[:, :2]
    heading = term.robot.data.heading_w

    goal_xy = torch.tensor([goal_x, goal_y], device=env.unwrapped.device, dtype=root_xy.dtype).unsqueeze(0)
    pos_error = torch.linalg.norm(root_xy - goal_xy, dim=1)

    goal_yaw_t = torch.full_like(heading, fill_value=goal_yaw)
    yaw_error = torch.abs(_wrap_to_pi(heading - goal_yaw_t))
    return pos_error, yaw_error


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with fixed-goal command injection for Spot navigation."""
    goal_x, goal_y, goal_yaw = _prompt_goal_from_stdin()

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Start the robot at origin, stationary.
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
        env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_goal"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    dt = env.unwrapped.step_dt

    # reset environment and lock the fixed command before first policy inference
    obs = env.get_observations()
    _set_fixed_goal(env, goal_x, goal_y, goal_yaw)

    print(
        "[INFO] Fixed goal command: "
        f"x={goal_x:.3f}, y={goal_y:.3f}, yaw={goal_yaw:.3f} rad, "
        f"pos_th={args_cli.pos_thresh:.3f}, yaw_th={args_cli.yaw_thresh:.3f}"
    )

    timestep = 0
    arrived_once = False
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            _set_fixed_goal(env, goal_x, goal_y, goal_yaw)
            pos_error, yaw_error = _compute_goal_error(env, goal_x, goal_y, goal_yaw)
            arrived = (pos_error < args_cli.pos_thresh) & (yaw_error < args_cli.yaw_thresh)

            if torch.any(arrived):
                actions = torch.zeros((env.num_envs, env.num_actions), device=env.unwrapped.device)
                if not arrived_once:
                    print(
                        f"[INFO] Arrival detected at step={timestep}: "
                        f"pos_err={pos_error[0].item():.3f} m, yaw_err={yaw_error[0].item():.3f} rad"
                    )
                    arrived_once = True
            else:
                actions = policy(obs)

            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        if timestep % 100 == 0:
            print(
                f"[INFO] step={timestep:05d} pos_err={pos_error[0].item():.3f} m "
                f"yaw_err={yaw_error[0].item():.3f} rad arrived={bool(arrived[0].item())}"
            )

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
