# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained BC policy for Spot locomotion.

This script loads a robomimic BC policy checkpoint and runs it in the Spot locomotion
environment for evaluation. The policy observes the current state (base_lin_vel,
base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel) and
outputs 12D joint-position actions.

Args:
    task: Name of the environment (default: Isaac-Velocity-Flat-Spot-v0)
    checkpoint: Path to the robomimic BC policy checkpoint (.pth file)
    num_envs: Number of environments to simulate
    num_rollouts: Number of evaluation rollouts
    horizon: Steps per rollout
    video: Record video during evaluation
    video_length: Length of the recorded video in steps
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic BC policy for Spot locomotion.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Spot-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the robomimic BC policy checkpoint (.pth file).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to simulate.",
)
parser.add_argument(
    "--num_rollouts",
    type=int,
    default=5,
    help="Number of evaluation rollouts to perform.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=500,
    help="Maximum steps per rollout.",
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record video during evaluation.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=101,
    help="Random seed.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras when recording video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import os
import random

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
from gymnasium.wrappers import RecordVideo

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# Observation term names expected by the BC policy (must match bc_low_dim.json low_dim list)
BC_OBS_TERM_NAMES = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
]

# Per-term dimensions for slicing the flat 48D observation vector
# Note: the flat obs is 48D; "actions" (12D) is the last term but NOT fed to the BC policy.
BC_OBS_TERM_DIMS = [3, 3, 3, 3, 12, 12]


def build_policy_obs(flat_obs: torch.Tensor) -> dict:
    """Build per-term observation dict from a flat observation tensor.

    The Spot flat env produces a 48D concatenated obs vector:
      [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
       velocity_commands(3), joint_pos(12), joint_vel(12), actions(12)]

    The BC policy was trained on the first 36D (excluding the last 'actions' term).

    Args:
        flat_obs: Flat observation tensor of shape [num_envs, 48] or [48].

    Returns:
        Dict mapping term name -> numpy array of shape [dim] (squeezed for robomimic).
    """
    if flat_obs.dim() == 2:
        flat_obs = flat_obs[0]  # take first env, shape [48]

    obs_dict = {}
    start = 0
    for name, dim in zip(BC_OBS_TERM_NAMES, BC_OBS_TERM_DIMS):
        obs_dict[name] = flat_obs[start : start + dim].cpu().numpy()
        start += dim

    return obs_dict


def rollout(policy, env, horizon: int, device: str) -> dict:
    """Perform a single evaluation rollout.

    Args:
        policy: Robomimic policy (callable with obs dict, implements start_episode()).
        env: Isaac Lab gymnasium environment (unwrapped ManagerBasedRLEnv).
        horizon: Maximum number of steps per rollout.
        device: Torch device string.

    Returns:
        Dict with keys: 'total_steps', 'terminated', 'truncated'.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()

    # obs_dict["policy"] is the flat concatenated [num_envs, 48] tensor
    flat_obs = obs_dict["policy"]

    total_steps = 0
    terminated = False
    truncated = False

    # Record initial velocity command (indices 9~11 in the 48D flat obs)
    # [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3), ...]
    init_vel_cmd = flat_obs[0, 9:12].cpu().numpy() if flat_obs.dim() == 2 else flat_obs[9:12].cpu().numpy()

    for step in range(horizon):
        # Build per-term obs dict for robomimic
        obs_for_policy = build_policy_obs(flat_obs)

        # Robomimic policy returns numpy array of shape [action_dim]
        action_np = policy(obs_for_policy)

        # Convert to torch tensor [1, 12] and move to device
        action = torch.from_numpy(action_np).to(device=device).view(1, env.action_space.shape[1])

        # Step environment
        obs_dict, reward, terminated_t, truncated_t, _ = env.step(action)
        flat_obs = obs_dict["policy"]

        total_steps += 1
        terminated = bool(terminated_t[0]) if hasattr(terminated_t, "__len__") else bool(terminated_t)
        truncated = bool(truncated_t[0]) if hasattr(truncated_t, "__len__") else bool(truncated_t)

        if terminated or truncated:
            break

    return {
        "total_steps": total_steps,
        "terminated": terminated,
        "truncated": truncated,
        "init_vel_cmd": init_vel_cmd,  # [vx, vy, wz]
    }


def main():
    """Run BC policy evaluation for Spot locomotion."""
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Set random seeds
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Keep observations as flat concatenated vector (48D for Spot)
    env_cfg.observations.policy.concatenate_terms = True
    # Disable observation noise for clean evaluation
    env_cfg.observations.policy.enable_corruption = False

    # Disable recorders
    env_cfg.recorders = None

    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    gym_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Wrap for video recording if requested
    if args_cli.video:
        log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        video_dir = os.path.join(log_dir, "videos", "bc_eval")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {video_dir}")
        gym_env = RecordVideo(gym_env, **video_kwargs)

    env = gym_env.unwrapped

    print(f"[INFO] Loading BC policy from: {args_cli.checkpoint}")
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    print(f"[INFO] Running {args_cli.num_rollouts} evaluation rollouts (horizon={args_cli.horizon}) ...")

    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Trial {trial + 1}/{args_cli.num_rollouts} ...")
        result = rollout(policy, env, args_cli.horizon, str(device))
        results.append(result)
        vc = result["init_vel_cmd"]
        print(
            f"[INFO] Trial {trial + 1}: steps={result['total_steps']}, "
            f"terminated={result['terminated']}, truncated={result['truncated']} | "
            f"vel_cmd=[vx={vc[0]:.2f}, vy={vc[1]:.2f}, wz={vc[2]:.2f}]"
        )

    # Summary statistics
    avg_steps = sum(r["total_steps"] for r in results) / len(results)
    print(f"\n[SUMMARY] {args_cli.num_rollouts} rollouts completed.")
    print(f"[SUMMARY] Average steps per rollout: {avg_steps:.1f}")

    # Failure analysis
    threshold = 100
    failures = [r for r in results if r["total_steps"] < threshold]
    successes = [r for r in results if r["total_steps"] >= threshold]

    print(f"\n[ANALYSIS] Failures (<{threshold} steps): {len(failures)}/{len(results)}")
    if failures:
        fail_vx = np.mean([r["init_vel_cmd"][0] for r in failures])
        fail_vy = np.mean([r["init_vel_cmd"][1] for r in failures])
        fail_wz = np.mean([r["init_vel_cmd"][2] for r in failures])
        print(f"[ANALYSIS]   Avg vel_cmd at failure: vx={fail_vx:.2f}, vy={fail_vy:.2f}, wz={fail_wz:.2f}")

    print(f"[ANALYSIS] Successes (>={threshold} steps): {len(successes)}/{len(results)}")
    if successes:
        succ_vx = np.mean([r["init_vel_cmd"][0] for r in successes])
        succ_vy = np.mean([r["init_vel_cmd"][1] for r in successes])
        succ_wz = np.mean([r["init_vel_cmd"][2] for r in successes])
        print(f"[ANALYSIS]   Avg vel_cmd at success: vx={succ_vx:.2f}, vy={succ_vy:.2f}, wz={succ_wz:.2f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
