# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAgger (Dataset Aggregation) training script for Spot locomotion BC policy.

Implements the DAgger algorithm for iterative imitation learning:
  1. Train initial BC policy on the provided expert demonstration dataset.
  2. Repeat for num_dagger_iters:
     a. Roll out the current student (BC) policy in the environment.
     b. At every step, query the expert (RSL-RL policy) for the correct action.
     c. Save (state, expert_action) pairs to a new HDF5 file and aggregate.
     d. Re-train the BC policy on the full aggregated dataset.
  3. Save the final policy.

The BC policy is loaded via robomimic ``FileUtils.policy_from_checkpoint()``.
The expert policy is a trained RSL-RL OnPolicyRunner checkpoint.

Args:
    task: Gym task ID (default: Isaac-Velocity-Flat-Spot-v0)
    expert_checkpoint: Path to the RSL-RL expert policy checkpoint (.pt file)
    initial_dataset: Path to the initial HDF5 demonstration dataset
    bc_checkpoint: Path to a pre-trained BC policy checkpoint to start from.
                   If not provided, BC is trained from scratch on initial_dataset first.
    num_dagger_iters: Number of DAgger iterations (default: 5)
    rollout_demos: Number of rollout demos to collect per DAgger iteration (default: 100)
    demo_length: Maximum steps per rollout demo (default: 500)
    num_epochs: Number of BC training epochs per iteration (default: 500)
    output_dir: Root output directory for checkpoints and aggregated datasets
    algo: Robomimic algorithm name (default: bc)
    num_envs: Number of parallel environments for rollouts (default: 1)
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Argument parsing (must happen before AppLauncher)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DAgger training for Spot locomotion BC policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Spot-v0",
    help="Gym task ID.",
)
parser.add_argument(
    "--expert_checkpoint",
    type=str,
    required=True,
    help="Path to RSL-RL expert policy checkpoint (.pt file).",
)
parser.add_argument(
    "--initial_dataset",
    type=str,
    required=True,
    help="Path to the initial HDF5 expert demonstration dataset.",
)
parser.add_argument(
    "--bc_checkpoint",
    type=str,
    default=None,
    help=(
        "Optional path to a pre-trained BC robomimic checkpoint (.pth) to start from. "
        "If not given, BC is trained from scratch on --initial_dataset first."
    ),
)
parser.add_argument(
    "--num_dagger_iters",
    type=int,
    default=5,
    help="Number of DAgger iterations.",
)
parser.add_argument(
    "--rollout_demos",
    type=int,
    default=100,
    help="Number of student rollout demos to collect per DAgger iteration.",
)
parser.add_argument(
    "--demo_length",
    type=int,
    default=500,
    help="Maximum steps per rollout demo.",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=500,
    help="Number of BC training epochs per iteration.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./dagger_output",
    help="Root output directory for checkpoints and aggregated datasets.",
)
parser.add_argument(
    "--algo",
    type=str,
    default="bc",
    help="Robomimic algorithm name (default: bc).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments for student rollouts.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)

# Append AppLauncher args (headless, device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim (headless for training loop)
if not hasattr(args_cli, "headless"):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Remaining imports (after Isaac Sim launch)
# ---------------------------------------------------------------------------

import glob
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys

import gymnasium as gym
import h5py
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

installed_rsl_rl_version = metadata.version("rsl-rl-lib")

# ---------------------------------------------------------------------------
# Observation term constants  (must match collect_expert_demos.py)
# ---------------------------------------------------------------------------

# Full 48D obs (used by RSL-RL expert)
OBS_TERM_NAMES = [
    "base_lin_vel",       # 3D
    "base_ang_vel",       # 3D
    "projected_gravity",  # 3D
    "velocity_commands",  # 3D
    "joint_pos",          # 12D
    "joint_vel",          # 12D
    "actions",            # 12D  (previous action — stored in obs but NOT fed to BC)
]
OBS_TERM_DIMS = [3, 3, 3, 3, 12, 12, 12]

# BC policy only uses the first 36D (excludes the trailing "actions" term)
BC_OBS_TERM_NAMES = OBS_TERM_NAMES[:-1]   # drop "actions"
BC_OBS_TERM_DIMS  = OBS_TERM_DIMS[:-1]    # drop 12


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def slice_obs_vector(flat_obs: torch.Tensor) -> dict:
    """Split a flat 48D observation tensor into named per-term tensors.

    Args:
        flat_obs: Shape [num_envs, 48].

    Returns:
        Dict: term_name -> tensor [num_envs, dim].
    """
    obs_dict = {}
    start = 0
    for name, dim in zip(OBS_TERM_NAMES, OBS_TERM_DIMS):
        obs_dict[name] = flat_obs[:, start : start + dim]
        start += dim
    return obs_dict


def build_bc_obs(flat_obs: torch.Tensor) -> dict:
    """Build the observation dict expected by the BC robomimic policy.

    Slices the first 36D from the 48D flat obs (drops the trailing 'actions'
    term which is only used by the RSL-RL expert).

    Args:
        flat_obs: Shape [48] (single env, already squeezed).

    Returns:
        Dict: term_name -> numpy array of shape [dim].
    """
    obs_dict = {}
    start = 0
    for name, dim in zip(BC_OBS_TERM_NAMES, BC_OBS_TERM_DIMS):
        obs_dict[name] = flat_obs[start : start + dim].cpu().numpy()
        start += dim
    return obs_dict


def count_demos_in_hdf5(hdf5_path: str) -> int:
    """Return the number of demos currently stored in an HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        return len(f["data"].keys())


def append_hdf5(src_path: str, dst_path: str) -> int:
    """Append all demos from src into dst, using collision-free demo IDs.

    Args:
        src_path: Path to source HDF5 file (newly collected rollouts).
        dst_path: Path to destination / aggregated HDF5 file.

    Returns:
        Number of demos appended.
    """
    with h5py.File(src_path, "r") as src_f, h5py.File(dst_path, "r+") as dst_f:
        src_data = src_f["data"]
        dst_data = dst_f["data"]

        # Determine next demo id by inspecting existing keys
        existing_ids = []
        for key in dst_data.keys():
            if key.startswith("demo_"):
                try:
                    existing_ids.append(int(key.split("_")[1]))
                except ValueError:
                    pass
        next_id = max(existing_ids, default=-1) + 1

        appended = 0
        for demo_key in sorted(src_data.keys()):
            new_key = f"demo_{next_id}"
            src_data.copy(demo_key, dst_data, name=new_key)
            # Update total step counter
            num_samples = dst_data[new_key].attrs.get("num_samples", 0)
            dst_data.attrs["total"] = dst_data.attrs.get("total", 0) + num_samples
            next_id += 1
            appended += 1

    return appended


def build_expert_and_env(device: str):
    """Create the Isaac Lab environment and load the RSL-RL expert policy.

    Returns:
        Tuple of (rsl_rl_env, expert_policy, gym_env).
    """
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.concatenate_terms = True
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.recorders = None

    gym_env = gym.make(args_cli.task, cfg=env_cfg)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    for field in ("share_cnn_encoders",):
        if hasattr(agent_cfg.algorithm, field):
            delattr(agent_cfg.algorithm, field)

    rsl_rl_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(rsl_rl_env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.expert_checkpoint)
    expert_policy = runner.get_inference_policy(device=gym_env.unwrapped.device)

    return rsl_rl_env, expert_policy, gym_env


def collect_dagger_rollouts(
    rsl_rl_env: RslRlVecEnvWrapper,
    expert_policy,
    bc_policy,
    output_hdf5: str,
    start_demo_id: int,
) -> int:
    """Collect student rollouts with expert-labelled actions.

    For each step:
      - Student (BC) policy chooses the action to execute in the environment.
      - Expert (RSL-RL) policy labels the current state with the correct action.
      - The (observation, expert_action) pair is stored in the HDF5 file.

    Args:
        rsl_rl_env: RSL-RL wrapped environment.
        expert_policy: RSL-RL inference policy.
        bc_policy: Robomimic BC policy (callable).
        output_hdf5: Path for the new rollout HDF5 file.
        start_demo_id: First demo_id to use when writing episodes.

    Returns:
        Number of demos successfully collected.
    """
    device = rsl_rl_env.device

    file_handler = HDF5DatasetFileHandler()
    file_handler.create(output_hdf5, env_name=args_cli.task)

    demos_collected = 0
    current_demo_id = start_demo_id

    while demos_collected < args_cli.rollout_demos:
        print(
            f"[INFO] DAgger rollout {demos_collected + 1}/{args_cli.rollout_demos} "
            f"(demo_id={current_demo_id}) ..."
        )

        rsl_rl_env.reset()
        obs = rsl_rl_env.get_observations()

        episode = EpisodeData()

        # Prepare BC policy for a new episode
        bc_policy.start_episode()

        for _ in range(args_cli.demo_length):
            flat_obs = obs["policy"]  # [num_envs, 48]
            flat_obs_single = flat_obs[0]  # [48] — we always use env index 0

            # ----- Student action: used to step the environment -----
            bc_obs = build_bc_obs(flat_obs_single)
            action_student_np = bc_policy(bc_obs)  # numpy [12]
            action_student = (
                torch.from_numpy(action_student_np)
                .to(device=device)
                .view(1, -1)
                .expand(args_cli.num_envs, -1)
            )

            # ----- Expert action: used as supervision label -----
            with torch.inference_mode():
                action_expert = expert_policy(obs)  # [num_envs, 12]
            action_expert_single = action_expert[0]  # [12]

            # ----- Store (obs, expert_action) -----
            obs_dict = slice_obs_vector(flat_obs_single.unsqueeze(0))
            for term_name, term_value in obs_dict.items():
                episode.add(f"obs/{term_name}", term_value.squeeze(0).cpu())
            episode.add("actions", action_expert_single.cpu())

            # ----- Step environment with student action -----
            obs, _, dones, _ = rsl_rl_env.step(action_student)

            if dones[0]:
                break

        episode.pre_export()

        if not episode.is_empty():
            file_handler.write_episode(episode, demo_id=current_demo_id)
            file_handler.flush()
            demos_collected += 1
            current_demo_id += 1
            steps = len(episode.data.get("actions", []))
            print(f"[INFO] Rollout saved: {steps} steps.")
        else:
            print("[WARN] Empty rollout skipped.")

    file_handler.close()
    return demos_collected


def run_bc_training(dataset_path: str, output_dir: str, iteration: int) -> str:
    """Call robomimic train.py as a subprocess and return the checkpoint path.

    Args:
        dataset_path: Path to the aggregated HDF5 dataset.
        output_dir: Root output directory for this DAgger run.
        iteration: Current DAgger iteration index (used for naming).

    Returns:
        Path to the latest checkpoint .pth file produced by train.py.
    """
    train_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "robomimic",
        "train.py",
    )
    train_script = os.path.normpath(train_script)

    exp_name = f"dagger_iter_{iteration}"
    log_dir_arg = os.path.join(output_dir, "robomimic_logs")

    cmd = [
        sys.executable,
        train_script,
        "--task", args_cli.task,
        "--algo", args_cli.algo,
        "--dataset", dataset_path,
        "--name", exp_name,
        "--log_dir", log_dir_arg,
        "--epochs", str(args_cli.num_epochs),
    ]

    print(f"[INFO] Launching BC training (iter={iteration}):")
    print("      " + " ".join(cmd))

    subprocess.run(cmd, check=True)

    # Locate the checkpoint directory produced by robomimic.
    # robomimic writes: <output_dir>/<task>/<exp_name>/models/
    ckpt_search_root = os.path.join("./logs", log_dir_arg, args_cli.task)
    pattern = os.path.join(ckpt_search_root, "**", exp_name, "models", "*.pth")
    ckpt_files = sorted(glob.glob(pattern, recursive=True))

    if not ckpt_files:
        # Fallback: broader search
        pattern2 = os.path.join(ckpt_search_root, "**", "models", "*.pth")
        ckpt_files = sorted(glob.glob(pattern2, recursive=True))

    if not ckpt_files:
        raise FileNotFoundError(
            f"Could not find any .pth checkpoint after BC training at iter {iteration}. "
            f"Searched under: {ckpt_search_root}"
        )

    # Pick the last checkpoint (highest epoch number)
    latest_ckpt = ckpt_files[-1]
    print(f"[INFO] BC checkpoint (iter={iteration}): {latest_ckpt}")
    return latest_ckpt


# ---------------------------------------------------------------------------
# Main DAgger loop
# ---------------------------------------------------------------------------

def main():
    """Run the full DAgger training loop."""
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    device_str = str(device)

    # Resolve output directory
    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Prepare aggregated dataset (copy initial dataset)
    # ------------------------------------------------------------------
    aggregated_dataset = os.path.join(output_dir, "aggregated_dataset.hdf5")
    if not os.path.exists(aggregated_dataset):
        print(f"[INFO] Copying initial dataset to: {aggregated_dataset}")
        shutil.copyfile(args_cli.initial_dataset, aggregated_dataset)
    else:
        print(f"[INFO] Resuming with existing aggregated dataset: {aggregated_dataset}")

    initial_demo_count = count_demos_in_hdf5(aggregated_dataset)
    print(f"[INFO] Initial dataset contains {initial_demo_count} demos.")

    # ------------------------------------------------------------------
    # Step 1: Train BC on initial dataset (iteration 0) if no checkpoint given
    # ------------------------------------------------------------------
    if args_cli.bc_checkpoint is not None:
        current_bc_ckpt = args_cli.bc_checkpoint
        print(f"[INFO] Using provided BC checkpoint: {current_bc_ckpt}")
        start_iter = 1
    else:
        print("[INFO] === Iteration 0: training BC on initial dataset ===")
        current_bc_ckpt = run_bc_training(aggregated_dataset, output_dir, iteration=0)
        start_iter = 1

    # ------------------------------------------------------------------
    # Build environment and expert policy (keep alive for all iterations)
    # ------------------------------------------------------------------
    print("[INFO] Building environment and loading expert policy ...")
    rsl_rl_env, expert_policy, gym_env = build_expert_and_env(device_str)

    # Track next demo_id globally so IDs never clash across iterations
    next_demo_id = count_demos_in_hdf5(aggregated_dataset)

    # ------------------------------------------------------------------
    # Step 2: DAgger iterations
    # ------------------------------------------------------------------
    for dagger_iter in range(start_iter, args_cli.num_dagger_iters + 1):
        print(f"\n[INFO] === DAgger iteration {dagger_iter}/{args_cli.num_dagger_iters} ===")

        # ------ 2a. Load current BC policy ------
        print(f"[INFO] Loading BC student policy from: {current_bc_ckpt}")
        bc_policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=current_bc_ckpt, device=device, verbose=False
        )

        # ------ 2b. Collect rollouts (student executes, expert labels) ------
        rollout_hdf5 = os.path.join(output_dir, f"rollouts_iter_{dagger_iter}.hdf5")
        n_collected = collect_dagger_rollouts(
            rsl_rl_env=rsl_rl_env,
            expert_policy=expert_policy,
            bc_policy=bc_policy,
            output_hdf5=rollout_hdf5,
            start_demo_id=next_demo_id,
        )
        print(f"[INFO] Collected {n_collected} DAgger demos.")

        # ------ 2c. Aggregate: append new rollouts to cumulative dataset ------
        n_appended = append_hdf5(rollout_hdf5, aggregated_dataset)
        next_demo_id += n_appended
        total_demos = count_demos_in_hdf5(aggregated_dataset)
        print(f"[INFO] Aggregated dataset now contains {total_demos} demos.")

        # ------ 2d. Re-train BC on aggregated dataset ------
        print(f"[INFO] Re-training BC on aggregated dataset ({total_demos} demos) ...")
        current_bc_ckpt = run_bc_training(aggregated_dataset, output_dir, iteration=dagger_iter)

    # ------------------------------------------------------------------
    # Final: copy best checkpoint to output root
    # ------------------------------------------------------------------
    final_ckpt = os.path.join(output_dir, "final_bc_policy.pth")
    shutil.copyfile(current_bc_ckpt, final_ckpt)
    print(f"\n[INFO] DAgger complete.")
    print(f"[INFO] Final BC policy saved to: {final_ckpt}")
    print(f"[INFO] Aggregated dataset: {aggregated_dataset}")

    # Save a summary JSON
    summary = {
        "task": args_cli.task,
        "num_dagger_iters": args_cli.num_dagger_iters,
        "rollout_demos_per_iter": args_cli.rollout_demos,
        "demo_length": args_cli.demo_length,
        "num_epochs_per_iter": args_cli.num_epochs,
        "final_bc_policy": final_ckpt,
        "aggregated_dataset": aggregated_dataset,
        "total_demos": count_demos_in_hdf5(aggregated_dataset),
    }
    summary_path = os.path.join(output_dir, "dagger_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[INFO] Summary written to: {summary_path}")

    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
