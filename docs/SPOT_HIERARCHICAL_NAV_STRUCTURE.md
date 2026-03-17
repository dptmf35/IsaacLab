# Spot Hierarchical Navigation Structure

## Overview
Spot navigation is designed as a hierarchical controller that separates:
- **High-level decision**: where to go
- **Low-level control**: how to walk stably

This decomposition improves training stability and transferability in complex warehouse environments.

## High-Level Layer: Navigation Policy
Role: Generates motion intent to reach a target pose.

It learns goal-conditioned behavior such as:
- Moving toward the target position
- Reducing heading error near the goal
- Choosing smooth directional changes instead of unstable sharp turns

Output:
- Low-dimensional motion command passed to the low-level locomotion policy

## Low-Level Layer: Locomotion Policy
Role: Converts high-level motion intent into joint-level commands.

It handles:
- Balance maintenance
- Foot placement and gait rhythm
- Disturbance rejection and posture stabilization

Output:
- Joint actions applied to Spot in simulation

## Reward Function Design (High-Level)
The reward terms are designed to prioritize survival, goal reaching, and heading quality.

- **termination_penalty (-400.0)**: Applies a large penalty when the episode ends due to body-ground contact.  
  Practical interpretation: a strong “game over” signal that enforces anti-fall behavior.

- **position_tracking (0.5)** and **position_tracking_fine_grained (0.5)**: Rewards approaching the target and adds extra credit for precise final convergence.

- **orientation_tracking (-0.2)**: Penalizes heading misalignment relative to target orientation to prevent drifted or twisted arrivals.

## Block Diagram
```text
[Goal Pose (x, y, yaw)]
          |
          v
+------------------------------+
| High-Level Navigation Policy |
|  - where to go               |
|  - outputs motion intent     |
+------------------------------+
          |
          | velocity/heading command
          v
+------------------------------+
| Low-Level Locomotion Policy  |
|  - how to walk               |
|  - balance & gait control    |
+------------------------------+
          |
          | joint actions
          v
+------------------------------+
| Spot Robot in Warehouse Sim  |
+------------------------------+
          |
          | state feedback (vel, posture, progress)
          +-------------------------------> back to policies
```
