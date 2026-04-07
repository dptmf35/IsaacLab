# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect expert demonstrations for Spot locomotion using a trained RSL-RL policy.

This script loads a pre-trained RSL-RL locomotion policy and uses it as an expert to
collect demonstration trajectories, saving them in HDF5 format compatible with robomimic BC training.

Args:
    task: Gym task ID (default: Isaac-Velocity-Flat-Spot-v0)
    checkpoint: Path to the RSL-RL policy checkpoint (.pt file)
    num_demos: Number of demonstrations to collect
    demo_length: Number of steps per demonstration
    output: Output HDF5 file path
    num_envs: Number of parallel environments (default: 1 for clean demos)
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect expert demos for Spot locomotion BC training.")
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
    help="Path to the RSL-RL policy checkpoint (.pt file).",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=200,
    help="Number of demonstrations to collect.",
)
parser.add_argument(
    "--demo_length",
    type=int,
    default=500,
    help="Number of steps per demonstration.",
)
parser.add_argument(
    "--output",
    type=str,
    default="./datasets/spot_locomotion_demos.hdf5",
    help="Output HDF5 file path.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--min_vel_mag",
    type=float,
    default=0.0,
    help="Minimum velocity magnitude filter: only save demos where |vy| + |wz| >= this value.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import importlib.metadata as metadata

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

installed_rsl_rl_version = metadata.version("rsl-rl-lib")


# Observation term names in order matching the policy group in flat_env_cfg.py
OBS_TERM_NAMES = [
    "base_lin_vel",       # 3D   / 선속도
    "base_ang_vel",       # 3D   / 각속도 
    "projected_gravity",  # 3D   / 투영된 중력 벡터 
    "velocity_commands",  # 3D   / 속도 목표값 cmd 
    "joint_pos",          # 12D  / 12개 관절의 위치 값
    "joint_vel",          # 12D  / 12개 관절의 속도 값 
    "actions",            # 12D (previous action)  
]

# Per-term dimensions corresponding to OBS_TERM_NAMES
OBS_TERM_DIMS = [3, 3, 3, 3, 12, 12, 12]


def slice_obs_vector(flat_obs: torch.Tensor) -> dict:
    """Split a flat 48D observation vector into named per-term tensors.

    Args:
        flat_obs: Flat observation tensor of shape [num_envs, 48].

    Returns:
        Dict mapping observation term name -> tensor [num_envs, dim].
    """
    obs_dict = {}
    start = 0
    for name, dim in zip(OBS_TERM_NAMES, OBS_TERM_DIMS):
        obs_dict[name] = flat_obs[:, start : start + dim]
        start += dim
    return obs_dict


def collect_demos_parallel(
    policy,
    rsl_rl_env: RslRlVecEnvWrapper,
    demo_length: int,
    num_envs: int,
) -> list:
    """Collect one demonstration per environment in parallel.

    Each environment records independently until its own done signal or
    demo_length is reached. All envs are stepped together every tick so
    GPU utilization stays high regardless of when individual envs finish.

    Args:
        policy: Inference policy from OnPolicyRunner.get_inference_policy().
        rsl_rl_env: RSL-RL wrapped environment.
        demo_length: Maximum number of steps per demonstration.
        num_envs: Number of parallel environments to record from.

    Returns:
        List of EpisodeData objects, one per environment.
    """
    episodes = [EpisodeData() for _ in range(num_envs)]
    done_flags = [False] * num_envs

    obs = rsl_rl_env.get_observations()

    for _ in range(demo_length):
        with torch.inference_mode():
            actions = policy(obs)

        # Record step for each env that hasn't finished yet
        for i in range(num_envs):
            if done_flags[i]:
                continue
            flat_obs_single = obs["policy"][i]  # [48]
            action_single = actions[i]           # [12]

            obs_dict = slice_obs_vector(flat_obs_single.unsqueeze(0))
            for term_name, term_value in obs_dict.items():
                episodes[i].add(f"obs/{term_name}", term_value.squeeze(0).cpu())
            episodes[i].add("actions", action_single.cpu())

        # Step all envs together
        obs, _, dones, _ = rsl_rl_env.step(actions)

        # Mark envs that are done
        for i in range(num_envs):
            if dones[i]:
                done_flags[i] = True

        # Stop when all envs are done
        if all(done_flags):
            break

    for ep in episodes:
        ep.pre_export()
    return episodes


def collect_demo(
    policy,
    rsl_rl_env: RslRlVecEnvWrapper,
    demo_length: int,
    env_idx: int,
) -> EpisodeData:
    """Collect a single demonstration trajectory from one environment."""
    return collect_demos_parallel(policy, rsl_rl_env, demo_length, num_envs=1)[0]


def main():
    """Collect expert demonstrations and save to HDF5."""
    device = args_cli.device if args_cli.device is not None else "cuda:0"

    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Use flat concatenated observations for RSL-RL policy compatibility
    env_cfg.observations.policy.concatenate_terms = True
    # Disable noise for cleaner expert demonstrations
    env_cfg.observations.policy.enable_corruption = False

    # Disable built-in recorders (we record manually)
    env_cfg.recorders = None

    # Create gymnasium environment
    gym_env = gym.make(args_cli.task, cfg=env_cfg)

    # Load agent configuration from task registry
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device

    # Handle deprecated rsl_rl cfg fields (version compatibility)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    # Remove fields not supported by older rsl_rl versions
    for field in ("share_cnn_encoders",):
        if hasattr(agent_cfg.algorithm, field):
            delattr(agent_cfg.algorithm, field)

    # Wrap for RSL-RL
    rsl_rl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    # Build runner and load checkpoint
    runner = OnPolicyRunner(rsl_rl_env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=gym_env.unwrapped.device)

    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Collecting {args_cli.num_demos} demos of {args_cli.demo_length} steps each.")

    # Prepare output HDF5 file
    output_path = args_cli.output
    if not output_path.endswith(".hdf5"):
        output_path += ".hdf5"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    file_handler = HDF5DatasetFileHandler()
    file_handler.create(output_path, env_name=args_cli.task)

    num_envs = args_cli.num_envs
    demos_collected = 0

    while demos_collected < args_cli.num_demos:
        remaining = args_cli.num_demos - demos_collected
        active_envs = min(num_envs, remaining)

        print(f"[INFO] Collecting demos {demos_collected + 1}~{demos_collected + active_envs}/{args_cli.num_demos} "
              f"(using {active_envs} parallel envs) ...")

        # Reset all envs to start fresh episodes
        rsl_rl_env.reset()

        # Capture initial velocity commands from flat obs BEFORE collecting
        # flat obs layout: [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3), ...]
        init_obs = rsl_rl_env.get_observations()
        init_flat = init_obs["policy"]  # [num_envs, 48]
        init_vel_cmds = init_flat[:active_envs, 9:12].cpu()  # [active_envs, 3] — vx, vy, wz

        # Collect from all active envs in parallel (one episode per env)
        episodes = collect_demos_parallel(
            policy=policy,
            rsl_rl_env=rsl_rl_env,
            demo_length=args_cli.demo_length,
            num_envs=active_envs,
        )

        for i, episode in enumerate(episodes):
            if demos_collected >= args_cli.num_demos:
                break
            if episode.is_empty():
                print("[WARN] Empty episode skipped.")
                continue

            # Velocity command filter: |vy| + |wz| >= min_vel_mag
            if args_cli.min_vel_mag > 0.0:
                vc = init_vel_cmds[i]
                vel_mag = abs(float(vc[1])) + abs(float(vc[2]))
                if vel_mag < args_cli.min_vel_mag:
                    print(f"[SKIP] vel_mag={vel_mag:.2f} < {args_cli.min_vel_mag} (vy={float(vc[1]):.2f}, wz={float(vc[2]):.2f})")
                    continue

            file_handler.write_episode(episode, demo_id=demos_collected)
            file_handler.flush()
            demos_collected += 1
            actions_len = len(episode.data.get("actions", []))
            print(f"[INFO] Demo {demos_collected} saved ({actions_len} steps).")

    file_handler.close()
    print(f"[INFO] Saved {demos_collected} demos to: {output_path}")

    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
