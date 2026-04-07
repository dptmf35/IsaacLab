#!/usr/bin/env bash
# =============================================================================
# A1 Imitation Learning Pipeline
#
# Full pipeline: Expert Demo Collection → BC Pre-training → DAgger Refinement
#
# Expert checkpoint: logs/rsl_rl/unitree_a1_flat/2026-04-07_12-23-55/model_299.pt
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SPOT_IL_DIR="$ISAACLAB_ROOT/scripts/imitation_learning/spot_locomotion"

EXPERT_CKPT="$ISAACLAB_ROOT/logs/rsl_rl/unitree_a1_flat/2026-04-07_12-23-55/model_299.pt"
TASK="Isaac-Velocity-Flat-Unitree-A1-v0"
ALGO="bc_rnn"

DATASET_DIR="$ISAACLAB_ROOT/datasets/a1_locomotion"
DAGGER_OUTPUT_DIR="$ISAACLAB_ROOT/dagger_output/a1_locomotion"
INITIAL_DATASET="$DATASET_DIR/a1_flat_demos.hdf5"

mkdir -p "$DATASET_DIR"
mkdir -p "$DAGGER_OUTPUT_DIR"

# -------------------------
# Step 1: Collect Expert Demos
# -------------------------
echo "=============================="
echo "Step 1: Collecting Expert Demos"
echo "=============================="
"$ISAACLAB_ROOT/isaaclab.sh" -p "$SPOT_IL_DIR/collect_expert_demos.py" \
    --task "$TASK" \
    --checkpoint "$EXPERT_CKPT" \
    --num_demos 500 \
    --demo_length 500 \
    --output "$INITIAL_DATASET" \
    --num_envs 4 \
    --headless

echo "[Done] Dataset saved to: $INITIAL_DATASET"

# -------------------------
# Step 2: DAgger Training
# -------------------------
echo "=============================="
echo "Step 2: DAgger Training (BC-RNN + DAgger)"
echo "=============================="
"$ISAACLAB_ROOT/isaaclab.sh" -p "$SPOT_IL_DIR/dagger.py" \
    --task "$TASK" \
    --expert_checkpoint "$EXPERT_CKPT" \
    --initial_dataset "$INITIAL_DATASET" \
    --algo "$ALGO" \
    --num_dagger_iters 5 \
    --rollout_demos 200 \
    --demo_length 500 \
    --num_epochs 500 \
    --output_dir "$DAGGER_OUTPUT_DIR" \
    --num_envs 4 \
    --headless

echo "[Done] DAgger complete. Checkpoints in: $DAGGER_OUTPUT_DIR"
echo ""
echo "To evaluate the final policy:"
echo "  ./isaaclab.sh -p $SPOT_IL_DIR/play_bc.py \\"
echo "    --task $TASK \\"
echo "    --checkpoint $DAGGER_OUTPUT_DIR/iter_5/models/model_epoch_500.pth \\"
echo "    --num_envs 1 --num_rollouts 10 --horizon 500"
