#!/bin/bash
# v5 iter별 체크포인트 순차 평가

BASE="dagger_output/bc_rnn_v5/robomimic_logs/Isaac-Velocity-Flat-Spot-v0"
BC_INIT="logs/robomimic/Isaac-Velocity-Flat-Spot-v0/bc_low_dim_spot_locomotion/20260406151750/models/model_epoch_1000.pth"

echo "========== BC init =========="
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint $BC_INIT --num_rollouts 10 --horizon 1000

echo "========== DAgger iter 1 =========="
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint $BASE/dagger_iter_1/20260406163735/models/model_epoch_500.pth \
  --num_rollouts 10 --horizon 1000

echo "========== DAgger iter 2 =========="
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint $BASE/dagger_iter_2/20260406165727/models/model_epoch_500.pth \
  --num_rollouts 10 --horizon 1000

echo "========== DAgger iter 3 =========="
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint $BASE/dagger_iter_3/20260406171527/models/model_epoch_500.pth \
  --num_rollouts 10 --horizon 1000

echo "========== DAgger iter 4 =========="
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint $BASE/dagger_iter_4/20260406173532/models/model_epoch_500.pth \
  --num_rollouts 10 --horizon 1000

echo "========== DAgger iter 5 (final) =========="
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint $BASE/dagger_iter_5/20260406175827/models/model_epoch_500.pth \
  --num_rollouts 10 --horizon 1000
