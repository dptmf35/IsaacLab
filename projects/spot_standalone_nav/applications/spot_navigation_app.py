# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone Isaac Sim inference for Spot navigation with runtime goal commands.

Supports two high-level checkpoint formats:
1) Raw RSL-RL checkpoint: model_XXXX.pt
2) Exported JIT checkpoint: exported/policy.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os
import queue
import sys
import threading
from collections.abc import Callable

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Standalone Spot navigation inference in warehouse USD.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Navigation-FullWarehouse-Spot-Play-v0",
    help="Navigation task to run.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point key in gym registry.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to high-level navigation checkpoint (raw model_XXXX.pt or exported JIT policy.pt).",
)
parser.add_argument(
    "--checkpoint-type",
    type=str,
    default="auto",
    choices=["auto", "rsl", "jit"],
    help="Checkpoint loader type: auto, raw rsl checkpoint, or jit.",
)
parser.add_argument(
    "--low-level-policy",
    type=str,
    default=None,
    help="Path to Spot low-level JIT policy.pt. Overrides env default if set.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--pos-thresh", type=float, default=0.25, help="Arrival threshold for XY position error (meters).")
parser.add_argument("--yaw-thresh", type=float, default=0.25, help="Arrival threshold for yaw error (radians).")
parser.add_argument(
    "--waypoint-step",
    type=float,
    default=4.0,
    help="Maximum XY distance (meters) for each auto-generated waypoint segment.",
)
parser.add_argument(
    "--disable-timeout",
    action="store_true",
    default=False,
    help="Disable episode time-out termination (episodes can still reset on other termination terms).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

CAMERA_LOOKAT = (-8.0, 3.0, 0.0)
CAMERA_EYE = (-8.0, -10.0, 9.0)
SPAWN_X = CAMERA_LOOKAT[0]
SPAWN_Y = CAMERA_LOOKAT[1]
SPAWN_YAW = 0.0


def input_worker(cmd_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Receive runtime commands from terminal without blocking the simulation loop."""
    if not sys.stdin.isatty():
        print("[WARN] Non-interactive stdin detected. Runtime goal updates are disabled.")
        return

    print("[INPUT] Runtime commands:")
    print("[INPUT]   - First goal required: x y yaw   (example: 1.0 -2.5 0.0)")
    print("[INPUT]   - Cancel current goal: c")
    print("[INPUT]   - Stop now: stop")
    print("[INPUT]   - Quit app: quit")
    while not stop_event.is_set():
        try:
            raw = input("goal> ").strip()
        except EOFError:
            break
        if raw == "":
            continue
        token = raw.lower()
        if token in ("quit", "q", "exit"):
            cmd_queue.put(("quit", None))
            break
        if token in ("stop", "s"):
            cmd_queue.put(("stop", None))
            continue
        if token in ("cancel", "c"):
            cmd_queue.put(("cancel", None))
            continue
        parts = raw.split()
        if len(parts) != 3:
            print("[WARN] Use `x y yaw`, `c`, `stop`, or `quit`.")
            continue
        try:
            goal = (float(parts[0]), float(parts[1]), float(parts[2]))
            cmd_queue.put(("goal", goal))
        except ValueError:
            print("[WARN] Invalid numbers. Example: 1.0 -2.5 0.0")


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def wrap_to_pi_scalar(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def interpolate_yaw_shortest(start_yaw: float, end_yaw: float, alpha: float) -> float:
    delta = wrap_to_pi_scalar(end_yaw - start_yaw)
    return wrap_to_pi_scalar(start_yaw + alpha * delta)


def generate_waypoints(
    start: tuple[float, float, float], goal: tuple[float, float, float], max_step_xy: float
) -> list[tuple[float, float, float]]:
    """Generate linear XY waypoints with shortest-path yaw interpolation."""
    sx, sy, syaw = start
    gx, gy, gyaw = goal
    dist = math.hypot(gx - sx, gy - sy)
    if max_step_xy <= 0.0:
        return [goal]
    num_segments = max(1, int(math.ceil(dist / max_step_xy)))
    waypoints: list[tuple[float, float, float]] = []
    for i in range(1, num_segments + 1):
        alpha = i / num_segments
        wx = sx + alpha * (gx - sx)
        wy = sy + alpha * (gy - sy)
        wyaw = interpolate_yaw_shortest(syaw, gyaw, alpha)
        waypoints.append((wx, wy, wyaw))
    return waypoints


def set_fixed_goal(base_env, goal_x: float, goal_y: float, goal_yaw: float) -> None:
    """Overwrite `pose_command` term in world frame and block resampling."""
    term = base_env.command_manager.get_term("pose_command")
    term.pos_command_w[:, 0] = goal_x
    term.pos_command_w[:, 1] = goal_y
    term.pos_command_w[:, 2] = term.robot.data.default_root_state[:, 2]
    term.heading_command_w[:] = goal_yaw
    term.time_left[:] = 1.0e9


def compute_goal_error(base_env, goal_x: float, goal_y: float, goal_yaw: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env XY/yaw errors in world frame."""
    term = base_env.command_manager.get_term("pose_command")
    root_xy = term.robot.data.root_pos_w[:, :2]
    heading = term.robot.data.heading_w

    goal_xy = torch.tensor([goal_x, goal_y], device=base_env.device, dtype=root_xy.dtype).unsqueeze(0)
    pos_error = torch.linalg.norm(root_xy - goal_xy, dim=1)
    goal_yaw_t = torch.full_like(heading, fill_value=goal_yaw)
    yaw_error = torch.abs(wrap_to_pi(heading - goal_yaw_t))
    return pos_error, yaw_error


def get_robot_pose_world(base_env) -> tuple[float, float, float]:
    term = base_env.command_manager.get_term("pose_command")
    x = float(term.robot.data.root_pos_w[0, 0].item())
    y = float(term.robot.data.root_pos_w[0, 1].item())
    yaw = float(term.robot.data.heading_w[0].item())
    return x, y, yaw


def build_policy_from_jit(checkpoint_path: str, device: str) -> tuple[Callable, Callable | None]:
    """Load exported JIT policy and return (policy_fn, reset_fn)."""
    policy = torch.jit.load(checkpoint_path, map_location=device)
    policy.eval()

    def _policy(obs):
        return policy(obs["policy"])

    return _policy, None


def build_policy_from_rsl(env: RslRlVecEnvWrapper, agent_cfg, checkpoint_path: str) -> tuple[Callable, Callable | None]:
    """Load raw RSL-RL checkpoint and return (policy_fn, reset_fn)."""
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    def _reset_fn(dones: torch.Tensor):
        policy_nn.reset(dones)

    return policy, _reset_fn


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.disable_fabric or args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # Set viewer camera for better warehouse visibility.
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = CAMERA_EYE
        env_cfg.viewer.lookat = CAMERA_LOOKAT

    # Force deterministic spawn: robot starts at (0, 0, yaw=0), zero velocity.
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (SPAWN_X, SPAWN_X),
            "y": (SPAWN_Y, SPAWN_Y),
            "yaw": (SPAWN_YAW, SPAWN_YAW),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    if args_cli.low_level_policy is not None:
        env_cfg.actions.pre_trained_policy_action.policy_path = os.path.abspath(args_cli.low_level_policy)

    if args_cli.disable_timeout and hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
        # guard: even if a timeout term survives in custom cfg, set effectively infinite episode length
        env_cfg.episode_length_s = 1.0e9
        print("[INFO] Disabled time_out termination for standalone run.")

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)

    print(f"[INFO] Creating standalone Isaac Sim environment for task: {args_cli.task}")
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    policy_fn = None
    reset_fn = None

    if args_cli.checkpoint_type in ("jit", "auto"):
        try:
            policy_fn, reset_fn = build_policy_from_jit(checkpoint_path, args_cli.device)
            print("[INFO] Loaded checkpoint as JIT policy.")
        except Exception as jit_exc:
            if args_cli.checkpoint_type == "jit":
                raise RuntimeError(
                    "Failed to load JIT checkpoint. Expected exported/policy.pt (TorchScript)."
                ) from jit_exc
            print(f"[INFO] JIT load failed ({type(jit_exc).__name__}). Falling back to raw RSL checkpoint...")

    if policy_fn is None:
        policy_fn, reset_fn = build_policy_from_rsl(env, agent_cfg, checkpoint_path)
        print("[INFO] Loaded checkpoint as raw RSL-RL model checkpoint.")

    obs = env.get_observations()
    goal_x, goal_y, goal_yaw = SPAWN_X, SPAWN_Y, SPAWN_YAW
    set_fixed_goal(base_env, goal_x, goal_y, goal_yaw)
    print(
        "[INFO] Robot spawn is fixed at "
        f"x={SPAWN_X:.3f}, y={SPAWN_Y:.3f}, yaw={SPAWN_YAW:.3f} "
        "and will hold until first goal is entered."
    )
    print(f"[INFO] Viewer eye={CAMERA_EYE}, lookat={CAMERA_LOOKAT}")
    print(f"[INFO] Arrival thresholds: pos_th={args_cli.pos_thresh:.3f}, yaw_th={args_cli.yaw_thresh:.3f}")

    step_count = 0
    arrived_once = False
    stop_motion = True
    goal_initialized = False
    waypoint_queue: list[tuple[float, float, float]] = []
    cmd_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(target=input_worker, args=(cmd_queue, stop_event), daemon=True)
    thread.start()

    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                while True:
                    try:
                        cmd, payload = cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "quit":
                        print("[INFO] Quit command received. Closing simulation.")
                        return
                    if cmd == "stop":
                        stop_motion = True
                        arrived_once = False
                        print("[INFO] Stop command received. Holding zero action.")
                        continue
                    if cmd == "cancel":
                        waypoint_queue.clear()
                        goal_initialized = False
                        stop_motion = True
                        arrived_once = False
                        print("[INFO] Goal canceled. Holding at current pose until a new goal is entered.")
                        continue
                    if cmd == "goal":
                        current_pose = get_robot_pose_world(base_env)
                        target_goal = payload
                        waypoint_queue = generate_waypoints(current_pose, target_goal, args_cli.waypoint_step)
                        goal_x, goal_y, goal_yaw = waypoint_queue.pop(0)
                        goal_initialized = True
                        stop_motion = False
                        arrived_once = False
                        print(
                            "[INFO] New goal accepted. "
                            f"waypoints={1 + len(waypoint_queue)} first=({goal_x:.3f}, {goal_y:.3f}, {goal_yaw:.3f})"
                        )

                set_fixed_goal(base_env, goal_x, goal_y, goal_yaw)
                pos_error, yaw_error = compute_goal_error(base_env, goal_x, goal_y, goal_yaw)
                arrived = (pos_error < args_cli.pos_thresh) & (yaw_error < args_cli.yaw_thresh)

                if torch.any(arrived) and goal_initialized and (not stop_motion) and len(waypoint_queue) > 0:
                    goal_x, goal_y, goal_yaw = waypoint_queue.pop(0)
                    arrived = torch.zeros_like(arrived, dtype=torch.bool)
                    print(
                        "[INFO] Waypoint reached. Next waypoint: "
                        f"x={goal_x:.3f}, y={goal_y:.3f}, yaw={goal_yaw:.3f} "
                        f"(remaining={len(waypoint_queue)})"
                    )

                if (not goal_initialized) or stop_motion or torch.any(arrived):
                    actions = torch.zeros(env.num_envs, env.num_actions, device=base_env.device)
                    if (not stop_motion) and (not arrived_once):
                        print(
                            f"[INFO] Arrival detected at step={step_count}: "
                            f"pos_err={pos_error[0].item():.3f} m, yaw_err={yaw_error[0].item():.3f} rad"
                        )
                        arrived_once = True
                        goal_initialized = False
                        stop_motion = True
                else:
                    actions = policy_fn(obs)

                obs, _, dones, _ = env.step(actions)
                if reset_fn is not None:
                    reset_fn(dones)

                if step_count % 100 == 0:
                    print(
                        f"[INFO] step={step_count:05d} pos_err={pos_error[0].item():.3f} m "
                        f"yaw_err={yaw_error[0].item():.3f} rad arrived={bool(arrived[0].item())} "
                        f"stop={stop_motion} waiting_goal={not goal_initialized} remaining_wp={len(waypoint_queue)}"
                    )
                step_count += 1
    finally:
        stop_event.set()
        thread.join(timeout=0.5)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
