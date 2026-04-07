# Unitree A1 Imitation Learning Pipeline

## 개요

Unitree A1 사족보행 로봇에 대한 모방학습(IL) 파이프라인.
RSL-RL PPO expert policy → 데모 수집 → BC-RNN pre-training → DAgger iterative refinement.

---

## 전체 흐름

```
[PPO Expert]  model_299.pt
      │
      ▼
[Step 1] collect_expert_demos.py  ──►  a1_flat_demos.hdf5
      │
      ▼
[Step 2] dagger.py  (BC-RNN 학습 + 반복 롤아웃 레이블링)
      │   iter 0: BC pre-train on expert demos
      │   iter 1~5: student rollout → expert label → 데이터 추가 → BC 재학습
      ▼
[Step 3] play_bc.py  (평가)
```

---

## 파일 구조

```
IsaacLab/
├── logs/rsl_rl/unitree_a1_flat/
│   └── 2026-04-07_12-23-55/
│       └── model_299.pt                          ← PPO Expert checkpoint
│
├── source/isaaclab_tasks/.../config/a1/
│   ├── flat_env_cfg.py                           ← 환경 설정 (Flat terrain)
│   ├── rough_env_cfg.py                          ← 환경 설정 (Rough terrain)
│   ├── __init__.py                               ← Gym 등록
│   └── agents/
│       ├── rsl_rl_ppo_cfg.py                     ← PPO 하이퍼파라미터
│       └── robomimic/
│           └── bc_rnn_low_dim.json               ← BC-RNN 하이퍼파라미터 (새로 추가)
│
├── datasets/a1_locomotion/
│   └── a1_flat_demos.hdf5                        ← 수집된 expert 데모
│
├── dagger_output/a1_locomotion/
│   ├── iter_0/                                   ← BC pre-train
│   ├── iter_1/ ~ iter_5/                         ← DAgger iterations
│   └── iter_5/models/model_epoch_500.pth         ← 최종 BC policy
│
└── scripts/imitation_learning/
    ├── spot_locomotion/                          ← 실제 스크립트 (A1도 재사용)
    │   ├── collect_expert_demos.py
    │   ├── dagger.py
    │   └── play_bc.py
    └── a1_locomotion/
        ├── run_il_pipeline.sh                    ← 전체 파이프라인 한 번에 실행
        └── A1_IL_PIPELINE.md                     ← 이 문서
```

---

## 환경 설정 (A1 Flat)

**파일**: `config/a1/flat_env_cfg.py`

```python
class UnitreeA1FlatEnvCfg(UnitreeA1RoughEnvCfg):
    # Rough 환경에서 상속, Flat용으로 오버라이드
    terrain_type = "plane"          # 평지
    height_scanner = None           # 높이 스캐너 없음
    terrain_curriculum = None       # 커리큘럼 없음
    flat_orientation_l2.weight = -2.5
    feet_air_time.weight = 0.25
```

**관찰 벡터 (48D)**:

| 항목 | 차원 | 설명 |
|------|------|------|
| `base_lin_vel` | 3 | 베이스 선속도 (x, y, z) |
| `base_ang_vel` | 3 | 베이스 각속도 (roll, pitch, yaw) |
| `projected_gravity` | 3 | 투영된 중력 벡터 |
| `velocity_commands` | 3 | 목표 속도 (vx, vy, wz) |
| `joint_pos` | 12 | 12개 관절 위치 |
| `joint_vel` | 12 | 12개 관절 속도 |
| `actions` | 12 | 이전 스텝 액션 |
| **합계** | **48** | RSL-RL expert 입력 |

> **BC는 48D 중 앞 36D만 사용** (마지막 `actions` 12D 제외)  
> 이유: BC는 recurrent이므로 이전 액션을 hidden state로 암묵적으로 기억함

**액션 벡터 (12D)**: 12개 관절의 target position (PPO action scale = 0.25)

---

## PPO Expert 설정

**파일**: `config/a1/agents/rsl_rl_ppo_cfg.py`

```python
class UnitreeA1FlatPPORunnerCfg:
    max_iterations = 300
    experiment_name = "unitree_a1_flat"
    
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims  = [128, 128, 128],   # Flat은 작은 네트워크
        critic_hidden_dims = [128, 128, 128],
        activation = "elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        clip_param       = 0.2,
        learning_rate    = 1e-3,
        schedule         = "adaptive",
        desired_kl       = 0.01,
        max_grad_norm    = 1.0,
        entropy_coef     = 0.01,
    )
```

> **주의**: RSL-RL 3.1.2에서 `share_cnn_encoders` 필드 오류 발생 가능  
> `train.py`와 `play.py`에 `delattr(agent_cfg.algorithm, 'share_cnn_encoders')` 패치 적용됨

---

## BC-RNN 설정

**파일**: `config/a1/agents/robomimic/bc_rnn_low_dim.json`

```json
{
  "algo": {
    "actor_layer_dims": [512, 512],       // MLP encoder (ReLU 활성화)
    "rnn": {
      "enabled": true,
      "hidden_dim": 400,                  // LSTM hidden state 크기
      "rnn_type": "LSTM",
      "num_layers": 2,                    // 2-layer stacked LSTM
      "horizon": 10                       // BPTT 길이
    },
    "gmm": {
      "enabled": false                    // MSE loss 사용 (GMM 비활성)
    },
    "loss": {
      "l2_weight": 1.0                    // MSE loss
    }
  },
  "train": {
    "seq_length": 10,
    "batch_size": 256,
    "num_epochs": 1000
  },
  "observation": {
    "low_dim": [
      "base_lin_vel", "base_ang_vel", "projected_gravity",
      "velocity_commands", "joint_pos", "joint_vel"
    ]
  }
}
```

**네트워크 구조**:
```
입력 (36D) → MLP [512 → 512] (ReLU) → LSTM [400 × 2layers] → MLP head → 출력 (12D)
                                           ↑
                                    hidden state (cross-step)
```

**GMM 비활성 이유**: RNN+GMM은 BPTT × NLL exp/log 연산으로 gradient explosion 발생  
(Policy_Grad_Norm 최대 195,000 관측 → `max_grad_norm=100` clipping으로도 불안정)

---

## Gym 등록

**파일**: `config/a1/__init__.py`

```python
gym.register(
    id="Isaac-Velocity-Flat-Unitree-A1-v0",
    kwargs={
        "env_cfg_entry_point": "...flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "robomimic_bc_rnn_cfg_entry_point": "...:robomimic/bc_rnn_low_dim.json",  # ← 추가
    },
)
```

---

## 실행 커맨드

### PPO 학습 (Expert 생성)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --headless
# checkpoint: logs/rsl_rl/unitree_a1_flat/<timestamp>/model_299.pt
```

### PPO 평가
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-A1-Play-v0 \
    --checkpoint logs/rsl_rl/unitree_a1_flat/2026-04-07_12-23-55/model_299.pt \
    --num_envs 1
```

### Step 1: Expert 데모 수집
```bash
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/collect_expert_demos.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --checkpoint logs/rsl_rl/unitree_a1_flat/2026-04-07_12-23-55/model_299.pt \
    --num_demos 500 \
    --demo_length 500 \
    --output datasets/a1_locomotion/a1_flat_demos.hdf5 \
    --num_envs 4 \
    --headless
```

### Step 2: DAgger 학습
```bash
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/dagger.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --expert_checkpoint logs/rsl_rl/unitree_a1_flat/2026-04-07_12-23-55/model_299.pt \
    --initial_dataset datasets/a1_locomotion/a1_flat_demos.hdf5 \
    --algo bc_rnn \
    --num_dagger_iters 5 \
    --rollout_demos 200 \
    --demo_length 500 \
    --num_epochs 500 \
    --output_dir dagger_output/a1_locomotion \
    --num_envs 4 \
    --headless
```

### Step 3: BC policy 평가
```bash
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
    --task Isaac-Velocity-Flat-Unitree-A1-v0 \
    --checkpoint dagger_output/a1_locomotion/iter_5/models/model_epoch_500.pth \
    --num_envs 1 --num_rollouts 10 --horizon 500
```

---

## DAgger 알고리즘 흐름

```
iter 0:
  BC.train(expert_demos)  →  student_policy_v0

iter k (k=1..5):
  for demo in range(rollout_demos):
    obs = env.reset()
    for t in range(demo_length):
      a_student = student_policy(obs)     ← 환경 스텝은 student로
      a_expert  = expert_policy(obs)      ← 레이블은 expert로
      dataset.add(obs, a_expert)          ← expert action으로 저장
      obs = env.step(a_student)
  
  aggregated_dataset += new_dataset
  BC.train(aggregated_dataset, epochs=500)
  student_policy = BC.checkpoint
```

**핵심 아이디어**: student가 방문하는 state distribution에서 expert action을 학습  
→ covariate shift 문제 해결

---

## 알려진 이슈 / 주의사항

| 문제 | 원인 | 해결 |
|------|------|------|
| `--algo bc` 사용 시 MLP로 학습됨 | 기본값이 bc (MLP) | `--algo bc_rnn` 명시 필수 |
| RNN+GMM gradient explosion | BPTT × NLL 연산 불안정 | GMM 비활성화 (MSE 사용) |
| `share_cnn_encoders` TypeError | RSL-RL 3.1.2 vs 5.0 불일치 | `train.py`, `play.py` 패치 적용됨 |
| BC output이 0에 수렴 (서있기) | MSE loss가 periodic action 평균화 | DAgger로 distribution shift 해결 |
