# Spot 사족보행 행동복제(BC) 파이프라인

Isaac Lab에서 Spot 로봇의 사족보행(Quadrupedal Locomotion) 정책을 행동복제(Behavioral Cloning)로 학습하는 완전한 파이프라인입니다. 사전 학습된 RSL-RL 전문가 정책으로부터 데모를 수집하고, robomimic을 사용하여 BC 정책을 훈련하며, 선택적으로 DAgger(Dataset Aggregation)로 반복 개선합니다.

## 개요

이 파이프라인의 목표는 RL로 학습한 전문가 정책의 행동을 모방하는 작은 신경망을 학습하는 것입니다.

**왜 필요한가:**
- **경량화**: RSL-RL 정책보다 훨씬 가볍고 빠른 추론
- **배포 용이**: 간단한 신경망으로 실제 로봇 배포 가능
- **데이터 효율**: 기존 RL 데이터를 활용한 효율적인 학습

**주요 단계:**
1. RSL-RL 전문가 정책에서 데모 수집
2. BC(행동복제) 정책 학습
3. BC 정책 평가
4. (선택) DAgger로 반복 개선

---

## 파이프라인 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    전문가 정책 (RSL-RL)                         │
│         사전 학습된 체크포인트 (*.pt 파일)                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  1. 데모 수집      │
        │ (collect_expert... │
        │      demos.py)     │
        └────────────┬───────┘
                     │
                     ▼
            ┌────────────────┐
            │   HDF5 데이터  │  ◄─── obs + actions
            │  spot_locomot  │
            │  ion_demos.    │
            │    hdf5        │
            └────────┬───────┘
                     │
         ┌───────────┴────────────┐
         │                        │
         ▼                        ▼
  ┌─────────────┐        ┌──────────────┐
  │ BC 학습     │        │ DAgger 훈련  │
  │(train.py)  │        │ (dagger.py)  │
  └──────┬──────┘        └──────┬───────┘
         │                      │
         ▼                      ▼
  ┌──────────────┐     ┌────────────────┐
  │ BC 정책      │     │ DAgger 정책    │
  │ (*.pth)      │     │ (*.pth)        │
  └──────┬───────┘     └────────┬───────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  2. 정책 평가    │
          │  (play_bc.py)    │
          └───────┬──────────┘
                  │
                  ▼
          ┌──────────────────┐
          │  평가 결과       │
          │  • 평균 스텝수   │
          │  • 성공/실패     │
          │  • 비디오(선택)  │
          └──────────────────┘
```

---

## 1. 데모 수집 (`collect_expert_demos.py`)

### 목적
RSL-RL 전문가 정책을 실행하여 상태-행동 쌍(trajectories)을 수집하고 HDF5 형식으로 저장합니다. 이 데이터는 BC 훈련의 기초가 됩니다.

### 구현 방법

#### 관찰값(Observation) 구조
Spot 환경에서 수집되는 48D 평탄 관찰값을 다음과 같이 분해합니다:

```
48D 평탄 관찰 = [
  base_lin_vel (3D),        # 기저부 선속도: [vx, vy, vz]
  base_ang_vel (3D),        # 기저부 각속도: [ωx, ωy, ωz]
  projected_gravity (3D),   # 투영된 중력 벡터 (body frame)
  velocity_commands (3D),   # 속도 목표값: [vx_cmd, vy_cmd, ωz_cmd]
  joint_pos (12D),          # 12개 관절의 위치값
  joint_vel (12D),          # 12개 관절의 속도값
  actions (12D)             # 이전 액션 (BC에서는 사용 안함)
]
```

**BC 훈련에 사용되는 관찰:** 첫 36D (actions 제외)

#### 수집 로직

```python
# 전문가 정책 실행
for step in range(demo_length):
    # 1. 현재 관찰에서 전문가 액션 선택
    action = expert_policy(obs)

    # 2. 단일 환경에서 관찰과 액션 추출
    flat_obs_single = obs["policy"][env_idx]      # [48]
    action_single = action[env_idx]               # [12]

    # 3. 관찰을 48D → 7개 항목으로 분해
    obs_dict = slice_obs_vector(flat_obs_single)

    # 4. HDF5에 저장 (actions 항목은 제외하고 저장)
    for term_name, term_value in obs_dict.items():
        if term_name != "actions":
            episode.add(f"obs/{term_name}", term_value)
    episode.add("actions", action_single)

    # 5. 환경 스텝
    obs, _, dones, _ = env.step(action)
```

#### HDF5 저장 구조

```
spot_locomotion_demos.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   ├── base_lin_vel          [steps, 3]
│   │   │   ├── base_ang_vel          [steps, 3]
│   │   │   ├── projected_gravity     [steps, 3]
│   │   │   ├── velocity_commands     [steps, 3]
│   │   │   ├── joint_pos             [steps, 12]
│   │   │   ├── joint_vel             [steps, 12]
│   │   │   └── actions               [steps, 12]  (actions는 저장됨)
│   │   └── (attributes): num_samples
│   ├── demo_1/
│   │   └── ...
│   └── demo_N/
└── metadata
    └── env_name: "Isaac-Velocity-Flat-Spot-v0"
```

### 사용법

#### CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--task` | `Isaac-Velocity-Flat-Spot-v0` | 환경 이름 |
| `--checkpoint` | (필수) | RSL-RL 체크포인트 경로 (`.pt` 파일) |
| `--num_demos` | 200 | 수집할 데모 개수 |
| `--demo_length` | 500 | 데모당 최대 스텝 수 |
| `--output` | `./datasets/spot_locomotion_demos.hdf5` | 출력 HDF5 파일 경로 |
| `--num_envs` | 1 | 병렬 환경 개수 (1권장: 깨끗한 데모) |
| `--disable_fabric` | False | Fabric 비활성화 |

#### 예시 명령어

```bash
# 1. Isaac Lab 환경 변수 설정
source ./isaaclab.sh

# 2. 기본 설정으로 데모 수집
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/collect_expert_demos.py \
  --checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt

# 3. 커스텀 설정으로 수집 (500개 데모, 각 1000 스텝)
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/collect_expert_demos.py \
  --checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt \
  --num_demos 500 \
  --demo_length 1000 \
  --output ./datasets/spot_demos_500.hdf5

# 4. 헤드리스 모드 (GUI 없음)
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/collect_expert_demos.py \
  --checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt \
  --headless
```

#### 출력 예시

```
[INFO] Loaded checkpoint: ./checkpoints/spot_locomotion_rsl_rl.pt
[INFO] Collecting 200 demos of 500 steps each.
[INFO] Collecting demo 1/200 ...
[INFO] Demo 1 saved (487 steps).
[INFO] Collecting demo 2/200 ...
[INFO] Demo 2 saved (493 steps).
...
[INFO] Collecting demo 200/200 ...
[INFO] Demo 200 saved (501 steps).
[INFO] Saved 200 demos to: ./datasets/spot_locomotion_demos.hdf5
```

---

## 2. BC 학습

### BC 정책 개요

행동복제는 감독 학습으로 전문가 정책의 행동을 직접 모방합니다:

```
관찰 x  ──[신경망]──> 예측 액션 ŷ
               ↑
               │ 손실 함수
               │ L = ||ŷ - y_expert||²
               │
         그라디언트 하강
```

### 두 가지 BC 설정 비교

#### 1. MLP BC (`bc_low_dim.json`)

**특징:**
- 단순 다층 퍼셉트론(MLP)
- 현재 관찰만 사용 (seq_length = 1)
- 빠르고 가볍지만 시간적 문맥 없음

**네트워크:**
```
입력 [36D] ─→ [512] ─→ [512] ─→ 출력 [12D]
         (ReLU)    (ReLU)    (Linear)
```

**주요 하이퍼파라미터:**
- `seq_length`: 1 (현재 프레임만)
- `actor_layer_dims`: [512, 512]
- `gmm.enabled`: False (단순 L2 손실)
- `rnn.enabled`: False
- 손실: L2 (MSE)

**적합한 경우:**
- 관찰만으로 결정 가능한 작업
- 배포 시 메모리/속도 중요
- 학습 데이터 충분

#### 2. BC-RNN (`bc_rnn_low_dim.json`)

**특징:**
- LSTM 기반 회귀신경망 + GMM
- 시간적 문맥 활용 (seq_length = 10)
- 동작이 복잡하거나 과거 정보 필요한 경우 우수

**네트워크:**
```
입력 시퀀스 [10 × 36D] ─→ LSTM [400hidden] × 2층 ─→ GMM (5개 모드) ─→ 출력 [12D]
                                                    ↓
                                              분포 샘플링
                                              (확률적 액션)
```

**주요 하이퍼파라미터:**
- `seq_length`: 10 (과거 10 프레임)
- `rnn.hidden_dim`: 400
- `rnn.num_layers`: 2
- `rnn.rnn_type`: "LSTM"
- `gmm.enabled`: True (5개 모드, softplus 활성화)
- `gmm.low_noise_eval`: True (평가 시 확정적)

**적합한 경우:**
- 멀티모달 행동 (여러 정책 옵션)
- 과거 상태가 중요
- 높은 정확도 필요

**비교:**
| 측면 | MLP BC | BC-RNN |
|------|--------|--------|
| 메모리 | 매우 적음 | 중간 |
| 추론 속도 | 매우 빠름 | 보통 |
| 정확도 | 보통 | 높음 |
| 학습 시간 | 빠름 | 느림 |
| 시간적 정보 | 없음 | 있음 (10 프레임) |

### BC 훈련

#### robomimic `train.py` 사용

robomimic 라이브러리의 훈련 스크립트를 사용합니다:

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 \
  --algo bc \
  --dataset ./datasets/spot_locomotion_demos.hdf5 \
  --config bc_low_dim.json \
  --name spot_bc_run1 \
  --log_dir ./logs/bc_training \
  --epochs 1000
```

#### 주요 훈련 설정 (bc_low_dim.json)

```json
{
  "train": {
    "num_data_workers": 4,        // 데이터 로더 스레드
    "hdf5_cache_mode": "all",     // HDF5 메모리 캐시
    "hdf5_use_swmr": true,        // 병렬 읽기
    "batch_size": 256,            // 배치 크기
    "num_epochs": 1000,           // 훈련 에포크
    "seed": 101
  },
  "algo": {
    "optim_params": {
      "policy": {
        "optimizer_type": "adam",
        "learning_rate": {
          "initial": 0.0001,
          "decay_factor": 0.1,
          "scheduler_type": "multistep"
        }
      }
    },
    "loss": {
      "l2_weight": 1.0            // L2(MSE) 손실 가중치
    },
    "actor_layer_dims": [512, 512] // 액터 신경망 크기
  }
}
```

#### CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--task` | (필수) | 환경 이름 |
| `--algo` | - | 알고리즘 (bc, bc_rnn 등) |
| `--config` | - | JSON 설정 파일 |
| `--dataset` | - | HDF5 데이터셋 경로 |
| `--name` | - | 실험 이름 |
| `--log_dir` | `./logs` | 체크포인트 저장 디렉토리 |
| `--epochs` | 1000 | 훈련 에포크 수 |

#### 예시 명령어

```bash
# MLP BC 훈련 (빠른 학습)
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 \
  --algo bc \
  --config bc_low_dim.json \
  --dataset ./datasets/spot_locomotion_demos.hdf5 \
  --name spot_bc_mlp \
  --log_dir ./logs/bc_training \
  --epochs 1000

# BC-RNN 훈련 (더 정확함)
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 \
  --algo bc \
  --config bc_rnn_low_dim.json \
  --dataset ./datasets/spot_locomotion_demos.hdf5 \
  --name spot_bc_rnn \
  --log_dir ./logs/bc_training \
  --epochs 1000
```

#### 훈련 출력

```
Epoch 1/1000 | Loss: 0.452 | lr: 0.0001
Epoch 2/1000 | Loss: 0.398 | lr: 0.0001
...
Epoch 500/1000 | Loss: 0.125 | lr: 0.00001
...
Epoch 1000/1000 | Loss: 0.089 | lr: 0.00001
[INFO] Training complete. Best checkpoint saved.
```

---

## 3. BC 정책 평가 (`play_bc.py`)

### 목적

학습된 BC 정책을 환경에서 실행하여 성능을 평가합니다. 정책의 일반화 능력과 안정성을 검증합니다.

### 구현 방법

#### 관찰 처리

BC 정책은 48D 평탄 관찰에서 36D만 추출하여 사용합니다:

```python
def build_policy_obs(flat_obs: torch.Tensor) -> dict:
    """
    48D 평탄 관찰 → 36D 관찰 딕셔너리

    입력: [48] = [base_lin_vel(3), base_ang_vel(3), ..., actions(12)]
    출력: {
        "base_lin_vel": [3],
        "base_ang_vel": [3],
        ...,
        "joint_vel": [12]
    }
    (actions 제외)
    """
    obs_dict = {}
    for name, dim in zip(BC_OBS_TERM_NAMES, BC_OBS_TERM_DIMS):
        obs_dict[name] = flat_obs[start:start+dim].cpu().numpy()
```

#### 평가 루프

```python
def rollout(policy, env, horizon: int, device: str) -> dict:
    policy.start_episode()              # 에피소드 초기화
    obs_dict, _ = env.reset()           # 환경 리셋
    flat_obs = obs_dict["policy"]       # [num_envs, 48]

    total_steps = 0
    for step in range(horizon):
        # 1. 관찰을 정책 입력 형식으로 변환
        obs_for_policy = build_policy_obs(flat_obs)

        # 2. 정책 추론
        action_np = policy(obs_for_policy)  # numpy [12]
        action = torch.from_numpy(action_np).to(device).view(1, 12)

        # 3. 환경 스텝
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        flat_obs = obs_dict["policy"]

        total_steps += 1

        if terminated or truncated:
            break

    return {"total_steps": total_steps, "terminated": terminated}
```

### 사용법

#### CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--task` | `Isaac-Velocity-Flat-Spot-v0` | 환경 이름 |
| `--checkpoint` | (필수) | BC 정책 체크포인트 (`.pth` 파일) |
| `--num_envs` | 1 | 병렬 환경 개수 |
| `--num_rollouts` | 5 | 평가 롤아웃 수 |
| `--horizon` | 500 | 롤아웃당 최대 스텝 |
| `--video` | False | 비디오 기록 |
| `--video_length` | 500 | 비디오 길이 (스텝) |
| `--seed` | 101 | 난수 시드 |

#### 예시 명령어

```bash
# 기본 평가 (5번의 500 스텝 롤아웃)
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint ./logs/bc_training/spot_bc_mlp/models/model_1000.pth

# 비디오 기록과 함께 평가
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint ./logs/bc_training/spot_bc_mlp/models/model_1000.pth \
  --num_rollouts 10 \
  --video \
  --video_length 500

# 더 긴 호라이즌으로 평가
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint ./logs/bc_training/spot_bc_rnn/models/model_1000.pth \
  --num_rollouts 20 \
  --horizon 1000 \
  --num_envs 4
```

#### 평가 출력

```
[INFO] Loading BC policy from: ./logs/bc_training/spot_bc_mlp/models/model_1000.pth
[INFO] Running 5 evaluation rollouts (horizon=500) ...
[INFO] Trial 1/5 ...
[INFO] Trial 1: steps=487, terminated=False, truncated=True
[INFO] Trial 2/5 ...
[INFO] Trial 2: steps=493, terminated=False, truncated=True
[INFO] Trial 3/5 ...
[INFO] Trial 3: steps=456, terminated=False, truncated=True
[INFO] Trial 4/5 ...
[INFO] Trial 4: steps=500, terminated=False, truncated=True
[INFO] Trial 5/5 ...
[INFO] Trial 5: steps=501, terminated=False, truncated=True

[SUMMARY] 5 rollouts completed.
[SUMMARY] Average steps per rollout: 487.4
```

**해석:**
- `average steps`: 정책이 얼마나 오래 안정적으로 움직이는지 나타냄
- 더 높을수록 좋음 (최대 호라이즌에 가까울수록 우수)
- `truncated=True`: 호라이즌 도달 (성공)
- `terminated=True`: 조기 종료 (실패)

---

## 4. DAgger (`dagger.py`)

### DAgger 알고리즘

**DAgger (Dataset Aggregation)**는 행동복제의 분포 시프트 문제를 해결합니다.

#### 분포 시프트 문제

행동복제는 기본 교사(전문가) 정책의 분포에서만 학습됩니다. 학생 정책의 오류가 쌓여서 상태가 훈련 분포를 벗어나면 성능이 급격히 떨어집니다.

```
                              시간 →
   상태 분포 시프트
   ╱
  ╱     학생의 실수
 ╱ ╱╱╱╱╱  쌓임
╱━━━━━━━━━━━━━━━━━━
     훈련 분포     분포 밖
     (안정)       (불안정)

BC만: 분포 밖의 상태에서 무너짐
```

#### DAgger 해결책

**반복적으로 데이터를 수집하고 학습합니다:**

```
반복 0: 전문가 → 데모 수집 → BC 학습
         ↓
반복 1: 학생 실행 → 전문가 레이블 → 데이터 누적 → BC 재학습
         ↓
반복 2: 학생 실행 → 전문가 레이블 → 데이터 누적 → BC 재학습
         ↓
...
반복 N: 최종 정책
```

**핵심: 학생 정책이 실행하는 상태에 대해 전문가가 정답을 제공**

### 구현 방법

#### Step 1: 초기 BC 훈련

```python
# 전문가 데모에서 BC 훈련
current_bc_ckpt = run_bc_training(
    dataset_path=initial_dataset,  # 전문가 데모
    output_dir=output_dir,
    iteration=0
)
```

#### Step 2: 반복 루프 (각 DAgger 반복)

각 반복에서:

```
1. 현재 BC 정책 로드
2. 학생 정책으로 롤아웃 수집
3. 데이터 누적
4. BC 재훈련
```

**2a. 학생 롤아웃 수집 + 전문가 레이블**

```python
def collect_dagger_rollouts(bc_policy, expert_policy, ...):
    for demo in range(num_demos):
        obs = env.reset()
        bc_policy.start_episode()

        for step in range(demo_length):
            # --- 학생 액션: 환경 스텝에 사용
            bc_obs = build_bc_obs(obs)
            action_student = bc_policy(bc_obs)  # [12]

            # --- 전문가 액션: 정답으로 저장
            action_expert = expert_policy(obs)  # [12]

            # --- 저장: (관찰, 전문가_액션)
            episode.add("obs/...", obs_terms)
            episode.add("actions", action_expert)  # ← 전문가 액션!

            # --- 환경 스텝: 학생 액션으로
            obs, _, done, _ = env.step(action_student)
```

**중요:** 환경은 학생 액션으로 스텝하지만, 저장되는 액션은 전문가 액션입니다.

**2b. 데이터 누적**

```python
# 기존 HDF5 파일에 새 데모 추가
append_hdf5(
    src_path="rollouts_iter_5.hdf5",  # 새 수집 데이터
    dst_path="aggregated_dataset.hdf5" # 누적 데이터셋
)

# 결과: 전문가(iter 0) + DAgger iter 1-5 = 총 데모 증가
```

HDF5 구조:
```
aggregated_dataset.hdf5
├── data/
│   ├── demo_0/         # 전문가 데모
│   ├── demo_1/
│   ├── ...
│   ├── demo_199/
│   ├── demo_200/       # DAgger iter 1 추가
│   ├── ...
│   ├── demo_299/
│   ├── demo_300/       # DAgger iter 2 추가
│   └── ...
```

**2c. BC 재훈련**

```python
current_bc_ckpt = run_bc_training(
    dataset_path=aggregated_dataset,  # 누적된 데이터
    output_dir=output_dir,
    iteration=dagger_iter
)
```

### 사용법

#### CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--task` | `Isaac-Velocity-Flat-Spot-v0` | 환경 이름 |
| `--expert_checkpoint` | (필수) | RSL-RL 전문가 체크포인트 |
| `--initial_dataset` | (필수) | 초기 HDF5 데모 |
| `--bc_checkpoint` | None | (선택) 시작할 BC 체크포인트 |
| `--num_dagger_iters` | 5 | DAgger 반복 횟수 |
| `--rollout_demos` | 100 | 각 반복당 수집 데모 수 |
| `--demo_length` | 500 | 데모당 최대 스텝 |
| `--num_epochs` | 500 | BC 훈련 에포크 |
| `--output_dir` | `./dagger_output` | 출력 디렉토리 |

#### 예시 명령어

```bash
# 기본 DAgger (5 반복, 100 데모/반복)
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/dagger.py \
  --expert_checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt \
  --initial_dataset ./datasets/spot_locomotion_demos.hdf5 \
  --output_dir ./dagger_output/run1

# 더 많은 반복과 데모
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/dagger.py \
  --expert_checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt \
  --initial_dataset ./datasets/spot_locomotion_demos.hdf5 \
  --num_dagger_iters 10 \
  --rollout_demos 200 \
  --demo_length 1000 \
  --output_dir ./dagger_output/run_long

# 기존 BC 정책에서 계속
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/dagger.py \
  --expert_checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt \
  --initial_dataset ./datasets/spot_locomotion_demos.hdf5 \
  --bc_checkpoint ./logs/bc_training/spot_bc_mlp/models/model_1000.pth \
  --num_dagger_iters 5 \
  --output_dir ./dagger_output/resume
```

#### DAgger 출력 구조

```
dagger_output/
├── aggregated_dataset.hdf5    # 누적된 데이터 (전문가 + DAgger)
├── rollouts_iter_1.hdf5       # DAgger iter 1 롤아웃
├── rollouts_iter_2.hdf5       # DAgger iter 2 롤아웃
├── rollouts_iter_3.hdf5       # ...
├── rollouts_iter_4.hdf5
├── rollouts_iter_5.hdf5
├── robomimic_logs/            # robomimic 훈련 로그
│   └── Isaac-Velocity-Flat-Spot-v0/
│       ├── dagger_iter_0/     # BC 초기 훈련 (iter 0)
│       ├── dagger_iter_1/     # BC 재훈련 (iter 1)
│       ├── ...
│       └── dagger_iter_5/     # 최종 BC 훈련
├── final_bc_policy.pth        # 최종 정책 (복사본)
└── dagger_summary.json        # 요약 정보

# dagger_summary.json 예시
{
  "task": "Isaac-Velocity-Flat-Spot-v0",
  "num_dagger_iters": 5,
  "rollout_demos_per_iter": 100,
  "demo_length": 500,
  "num_epochs_per_iter": 500,
  "final_bc_policy": "./dagger_output/run1/final_bc_policy.pth",
  "aggregated_dataset": "./dagger_output/run1/aggregated_dataset.hdf5",
  "total_demos": 700        # 초기 200 + (5 × 100)
}
```

#### DAgger 훈련 로그 예시

```
[INFO] === Iteration 0: training BC on initial dataset ===
[INFO] Loading BC student policy from: ./logs/bc_training/dagger_iter_0/models/model_500.pth

[INFO] === DAgger iteration 1/5 ===
[INFO] Loading BC student policy from: ./logs/bc_training/dagger_iter_0/models/model_500.pth
[INFO] DAgger rollout 1/100 (demo_id=200) ...
[INFO] Rollout saved: 487 steps.
[INFO] DAgger rollout 2/100 (demo_id=201) ...
[INFO] Rollout saved: 492 steps.
...
[INFO] Collected 100 DAgger demos.
[INFO] Aggregated dataset now contains 300 demos.
[INFO] Re-training BC on aggregated dataset (300 demos) ...
      python train.py --task Isaac-Velocity-Flat-Spot-v0 --algo bc \
        --dataset ./dagger_output/run1/aggregated_dataset.hdf5 \
        --name dagger_iter_1 --log_dir ./dagger_output/run1/robomimic_logs \
        --epochs 500
[INFO] BC checkpoint (iter=1): ./logs/robomimic_logs/Isaac-Velocity-Flat-Spot-v0/dagger_iter_1/models/model_500.pth

[INFO] === DAgger iteration 2/5 ===
[INFO] Loading BC student policy from: ./logs/robomimic_logs/.../dagger_iter_1/models/model_500.pth
...

[INFO] DAgger complete.
[INFO] Final BC policy saved to: ./dagger_output/run1/final_bc_policy.pth
[INFO] Aggregated dataset: ./dagger_output/run1/aggregated_dataset.hdf5
```

---

## 전체 실험 순서 (처음부터 끝까지)

### 전제 조건

```bash
# 1. Isaac Lab 설치 및 환경 구성
cd /home/yeseul/IsaacLab
./isaaclab.sh -i  # 확장 설치

# 2. 전문가 RSL-RL 정책 준비 (학습되었다고 가정)
# ./checkpoints/spot_locomotion_rsl_rl.pt 존재함
```

### 전체 파이프라인 실행

#### Phase 1: 데모 수집 (20분)

```bash
# 1단계: 전문가 RSL-RL로 200개 데모 수집
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/collect_expert_demos.py \
  --checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt \
  --num_demos 200 \
  --demo_length 500 \
  --output ./datasets/spot_locomotion_demos.hdf5 \
  --headless

# 출력: ./datasets/spot_locomotion_demos.hdf5 (약 500MB-1GB)
```

#### Phase 2a: BC 학습 - MLP 버전 (1시간)

```bash
# 2a단계: 간단한 MLP BC 훈련
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 \
  --algo bc \
  --config source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/agents/robomimic/bc_low_dim.json \
  --dataset ./datasets/spot_locomotion_demos.hdf5 \
  --name spot_bc_mlp \
  --log_dir ./logs/bc_training \
  --epochs 1000

# 출력: ./logs/bc_training/Isaac-Velocity-Flat-Spot-v0/spot_bc_mlp/models/
#       ├── model_100.pth
#       ├── model_200.pth
#       └── model_1000.pth
```

#### Phase 2b: BC 학습 - RNN 버전 (2시간)

```bash
# 2b단계: LSTM+GMM BC-RNN 훈련
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 \
  --algo bc \
  --config source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/agents/robomimic/bc_rnn_low_dim.json \
  --dataset ./datasets/spot_locomotion_demos.hdf5 \
  --name spot_bc_rnn \
  --log_dir ./logs/bc_training \
  --epochs 1000

# 출력: ./logs/bc_training/Isaac-Velocity-Flat-Spot-v0/spot_bc_rnn/models/model_1000.pth
```

#### Phase 3a: BC 평가 - MLP

```bash
# 3a단계: MLP BC 평가
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint ./logs/bc_training/Isaac-Velocity-Flat-Spot-v0/spot_bc_mlp/models/model_1000.pth \
  --num_rollouts 10 \
  --horizon 500 \
  --video

# 출력:
# [SUMMARY] Average steps per rollout: 487.3
# 비디오: ./logs/bc_training/.../videos/bc_eval/
```

#### Phase 3b: BC 평가 - RNN

```bash
# 3b단계: BC-RNN 평가
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint ./logs/bc_training/Isaac-Velocity-Flat-Spot-v0/spot_bc_rnn/models/model_1000.pth \
  --num_rollouts 10 \
  --horizon 500 \
  --video

# 출력:
# [SUMMARY] Average steps per rollout: 492.1
# 비디오: ./logs/bc_training/.../videos/bc_eval/
```

#### Phase 4: DAgger (선택, 3시간)

```bash
# 4단계: MLP BC로 DAgger 5회 반복
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/dagger.py \
  --expert_checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt \
  --initial_dataset ./datasets/spot_locomotion_demos.hdf5 \
  --bc_checkpoint ./logs/bc_training/Isaac-Velocity-Flat-Spot-v0/spot_bc_mlp/models/model_1000.pth \
  --num_dagger_iters 5 \
  --rollout_demos 100 \
  --demo_length 500 \
  --num_epochs 500 \
  --output_dir ./dagger_output/spot_bc_mlp \
  --headless

# 출력:
# ./dagger_output/spot_bc_mlp/
# ├── aggregated_dataset.hdf5  (700 demos)
# ├── final_bc_policy.pth
# └── dagger_summary.json
```

#### Phase 5: DAgger 최종 평가

```bash
# 5단계: DAgger 최종 정책 평가
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint ./dagger_output/spot_bc_mlp/final_bc_policy.pth \
  --num_rollouts 10 \
  --horizon 500 \
  --video

# 출력:
# [SUMMARY] Average steps per rollout: 495.8 (향상됨!)
```

### 실행 시간 요약

| 단계 | 시간 | 설명 |
|------|------|------|
| 데모 수집 | 20분 | 200 demo × 500 step |
| BC-MLP 훈련 | 1시간 | 1000 epochs, 256 batch |
| BC-RNN 훈련 | 2시간 | 더 큰 모델 |
| BC-MLP 평가 | 10분 | 10 rollouts × 500 step |
| BC-RNN 평가 | 10분 | 10 rollouts × 500 step |
| DAgger (5 iter) | 3시간 | 5 × (100 demo + 훈련) |
| **총계** | **6.5시간** | GPU 필수 |

---

## 실험 결과 및 팁

### BC-MLP vs BC-RNN vs DAgger 비교

다양한 조건에서 실제 성능을 비교하려면 같은 데이터와 설정으로 평가해야 합니다.

#### 이론적 성능 예상

```
정책          정확도   속도    메모리   추론시간
─────────────────────────────────────────────
BC-MLP        75%    빠름    적음    0.1ms
BC-RNN        88%    보통    중간    0.5ms
BC-RNN-DAgger  92%   보통    중간    0.5ms
전문가(RSL-RL) 100%   느림    큼      2ms
```

#### 평가 메트릭

1. **평균 스텝 수** (호라이즌에 대한)
   - 정책이 얼마나 오래 안정적으로 움직이는가
   - 500 step horizon: 450+ 좋음, 400-450 보통, <400 재훈련 필요

2. **성공률**
   - truncated (성공) vs terminated (실패) 비율

3. **속도**
   - 추론 시간 (정책이 얼마나 빠른가)
   - 배포 환경에서 중요

### 하이퍼파라미터 튜닝 팁

#### 데모 수집

```bash
# 데모 부족 시 (성능 < 70%)
--num_demos 500      # 200 → 500
--demo_length 1000   # 500 → 1000

# 메모리 부족 시
--num_envs 1         # 병렬화 감소
--num_demos 100      # 데모 감소

# 고품질 데모 필요 시
--num_envs 1         # 각각 깨끗하게 수집
```

#### BC 훈련

**MLP BC의 경우:**
```json
{
  "train": {
    "batch_size": 512,        // 증가: 더 안정적
    "num_epochs": 2000        // 증가: 더 수렴
  },
  "algo": {
    "actor_layer_dims": [1024, 1024]  // 확대: 더 복잡한 패턴 학습
  }
}
```

**BC-RNN의 경우:**
```json
{
  "train": {
    "seq_length": 20,         // 10 → 20: 더 긴 문맥
    "batch_size": 128         // 감소: 메모리 부족 시
  },
  "algo": {
    "rnn": {
      "hidden_dim": 512,      // 증가: 더 큰 상태
      "num_layers": 3         // 증가: 더 깊은 모델
    }
  }
}
```

#### DAgger

```bash
# 데이터 분포 시프트 심할 때
--num_dagger_iters 10    # 5 → 10: 더 많은 반복
--rollout_demos 200      # 100 → 200: 더 많은 데이터

# 빠른 수렴
--num_dagger_iters 3     # 5 → 3: 적은 반복
--rollout_demos 50       # 100 → 50: 적은 데이터
```

### 일반적인 문제 해결

#### 문제: BC 정책이 떨어짐 (early terminate)

**원인:**
- 데이터 부족
- 네트워크 너무 작음
- 학습률 너무 높음

**해결:**
```bash
# 1. 데모 증가
--num_demos 500

# 2. 네트워크 확대 (bc_low_dim.json)
"actor_layer_dims": [1024, 1024]

# 3. BC-RNN 사용 (시간적 정보)
--config bc_rnn_low_dim.json
```

#### 문제: 훈련 손실이 수렴하지 않음

**원인:**
- 학습률 너무 높음
- 배치 크기 너무 작음
- 데이터 품질 낮음

**해결:**
```bash
# bc_low_dim.json에서
"learning_rate": {
  "initial": 0.00005        # 0.0001 → 0.00005 (감소)
}
"batch_size": 512           # 256 → 512 (증가)
```

#### 문제: DAgger 성능 개선 안됨

**원인:**
- 반복 너무 적음
- 롤아웃 데모 너무 적음
- 초기 BC 정책 너무 약함

**해결:**
```bash
# 1. 반복 증가
--num_dagger_iters 10

# 2. 롤아웃 데모 증가
--rollout_demos 200

# 3. 초기 BC를 더 오래 훈련
# BC 훈련 시 --epochs 2000 (1000 → 2000)
```

### 배포 체크리스트

학습한 BC 정책을 실제 로봇에 배포할 때:

- [ ] **관찰 전처리 확인**: 48D → 36D (actions 제외) 추출
- [ ] **정규화 확인**: 관찰이 훈련 범위 내인지 확인
- [ ] **액션 클리핑**: 출력 액션이 [-1, 1] 범위인지 확인
- [ ] **프레임 스키핑**: 시뮬레이션과 실제 로봇의 시간 스케일 맞추기
- [ ] **노이즈 테스트**: 센서 노이즈에 대한 견고성 검증
- [ ] **안전 제한**: 불안전한 행동에 대한 제약 추가

### 참고 자료

- **robomimic**: https://robomimic.github.io/
- **DAgger 원본 논문**: Kahn et al., "GAIL: Generative Adversarial Imitation Learning"
- **Isaac Lab**: https://github.com/isaac-sim/IsaacLab
- **RSL-RL**: https://github.com/leggedrobotics/rsl_rl

---

## 빠른 시작 템플릿

### 최소 구성 (10분 BC 실행)

```bash
# 1. 작은 데이터셋으로 빠른 테스트
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/collect_expert_demos.py \
  --checkpoint ./checkpoints/spot_locomotion_rsl_rl.pt \
  --num_demos 50 \
  --output ./datasets/test_demos.hdf5 \
  --headless

# 2. 빠른 훈련 (에포크 100만)
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Velocity-Flat-Spot-v0 \
  --algo bc \
  --config bc_low_dim.json \
  --dataset ./datasets/test_demos.hdf5 \
  --name test_bc \
  --log_dir ./logs \
  --epochs 100

# 3. 평가
./isaaclab.sh -p scripts/imitation_learning/spot_locomotion/play_bc.py \
  --checkpoint ./logs/Isaac-Velocity-Flat-Spot-v0/test_bc/models/model_100.pth
```

### 프로덕션 구성 (전체 파이프라인)

위 **전체 실험 순서** 섹션 참고.

---

## 자주 묻는 질문 (FAQ)

**Q: BC로는 전문가 수준의 성능을 얻을 수 없나요?**

A: BC 최고 정확도는 보통 전문가의 85-95% 수준입니다. 더 높은 성능이 필요하면 DAgger를 사용하세요.

**Q: 데모는 몇 개가 필요한가요?**

A: 최소 100-200개. 관찰 공간이 복잡하면 500+ 권장. DAgger는 초기 200개에서 시작 가능.

**Q: BC-RNN이 항상 MLP보다 나은가요?**

A: 아니오. MLP가 더 빠르고 가볍습니다. 시간적 정보가 중요하면 RNN 사용. 단순 작업이면 MLP 권장.

**Q: DAgger가 꼭 필요한가요?**

A: 초기 BC 성능이 70% 이상이면 DAgger 없이도 가능. 50% 이하면 DAgger 권장.

**Q: GPU가 필요한가요?**

A: 훈련에는 필수. 추론(play_bc.py)은 CPU에서도 가능.

**Q: 어떤 체크포인트를 사용해야 하나요?**

A: 마지막 에포크 (model_1000.pth)보다는 **최저 손실 에포크** 추천. robomimic은 자동으로 베스트 모델을 저장합니다.

