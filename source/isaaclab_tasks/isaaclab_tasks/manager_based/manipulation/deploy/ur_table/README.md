# UR Table 강화학습 환경

UR 로봇 암 + 흡착 그리퍼 + X축 프리즈매틱 테이블로 구성된 커스텀 Digital Twin USD 기반의 Isaac Lab RL 환경입니다.

---

## 목차

1. [디렉토리 구조](#1-디렉토리-구조)
2. [Isaac Lab 환경 구축 개요](#2-isaac-lab-환경-구축-개요)
3. [USD 에셋 및 로봇 설정](#3-usd-에셋-및-로봇-설정)
4. [MDP 설계 원칙 및 과정](#4-mdp-설계-원칙-및-과정)
   - 4.1 [Scene 구성](#41-scene-구성)
   - 4.2 [Action 설계](#42-action-설계)
   - 4.3 [Observation 설계](#43-observation-설계)
   - 4.4 [Command 설계](#44-command-설계)
   - 4.5 [Reward 설계](#45-reward-설계)
   - 4.6 [Termination 설계](#46-termination-설계)
   - 4.7 [Event (Reset) 설계](#47-event-reset-설계)
5. [태스크별 환경 설정](#5-태스크별-환경-설정)
   - 5.1 [Align 태스크 (경량, 권장 시작점)](#51-align-태스크-경량-권장-시작점)
   - 5.2 [Pick-and-Place 태스크](#52-pick-and-place-태스크)
6. [RSL-RL PPO 학습 설정](#6-rsl-rl-ppo-학습-설정)
7. [실행 방법](#7-실행-방법)
8. [코드 흐름 추적](#8-코드-흐름-추적)
9. [TODO](#9-todo)

---

## 1. 디렉토리 구조

```
ur_table/
├── ur_table_env_cfg.py             # 베이스 환경 설정 (모든 태스크 공통 MDP 골격)
├── mdp/
│   ├── __init__.py                 # 공통 + 태스크별 MDP 모듈 통합 import
│   ├── rewards.py                  # UR Table 전용 reward 함수 (align 등)
│   └── terminations.py             # UR Table 전용 termination 함수
├── config/
│   └── ur_robot_system/
│       ├── __init__.py             # Gym 환경 ID 등록
│       ├── joint_pos_env_cfg.py    # 로봇별 구체 설정 (USD 연결, 태스크별 서브클래스)
│       └── agents/
│           └── rsl_rl_ppo_cfg.py   # RSL-RL PPO Runner 설정
└── README.md
```

**MDP 모듈 의존 관계:**

```
ur_table/mdp/__init__.py
  ├── deploy/mdp                        # joint_pos, joint_vel, action_rate_l2, action_l2 등 공통 함수
  ├── pick_place_suction/mdp            # ee_pos_w, ee_quat_w, goal_pos_command, goal_quat_command,
  │                                     # suction_state, approach_box_reward 등 pick-place 전용 함수
  ├── ur_table/mdp/rewards.py           # ee_goal_pos_reward, ee_goal_ori_reward (align 전용)
  └── ur_table/mdp/terminations.py      # ee_at_goal (align 전용)
```

---

## 2. Isaac Lab 환경 구축 개요

### Isaac Lab의 Manager-Based 환경 패러다임

Isaac Lab은 두 가지 환경 패러다임을 제공합니다:

| 패러다임 | 특징 | 적합한 경우 |
|----------|------|-------------|
| **Manager-Based** | 컴포넌트(Manager)를 조합해 환경 구성, 설정 파일 중심 | 표준 조작/이동 태스크, 재사용성 중요할 때 |
| **Direct** | `ManagerBasedRLEnv`를 서브클래싱, 메서드 직접 오버라이드 | 커스텀 물리 로직, 특수한 렌더링이 필요할 때 |

이 환경은 **Manager-Based** 방식을 사용합니다. 환경의 각 구성 요소(Action, Observation, Reward, Termination 등)를 독립적인 Manager 클래스가 담당하며, 이를 `@configclass`로 선언된 설정 파일에서 조합합니다.

### 환경 클래스 계층 구조

```
ManagerBasedRLEnvCfg          ← Isaac Lab 베이스 (라이브러리)
    └── URTableEnvCfg          ← 태스크 공통 골격 (ur_table_env_cfg.py)
            ├── URRobotSystemEnvCfg        ← 로봇별 구체 설정 (joint_pos_env_cfg.py)
            │       └── URRobotSystemEnvCfg_PLAY
            ├── URTableAlignEnvCfg         ← Align 태스크 오버라이드
            │       └── URTableAlignEnvCfg_PLAY
            └── (향후) URTablePickPlaceEnvCfg
```

**설계 원칙:** 공통 Scene·Action·Observation 구조는 베이스(`URTableEnvCfg`)에 정의하고, 로봇 USD 연결과 태스크별 Reward·Termination만 서브클래스에서 오버라이드합니다. 이렇게 하면 새로운 태스크를 추가할 때 변경되는 부분만 명시적으로 선언할 수 있습니다.

### `@configclass`와 `MISSING`

Isaac Lab의 모든 설정은 `@configclass` 데코레이터로 선언된 dataclass입니다.

```python
@configclass
class URTableEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=256, env_spacing=3.0)
    rewards: RewardsCfg = RewardsCfg()
```

- `MISSING`으로 표시된 필드는 서브클래스의 `__post_init__`에서 반드시 채워야 합니다.
- `__post_init__`에서 `super().__post_init__()`을 먼저 호출한 후 필드를 덮어씁니다.
- 설정 객체를 생성자 키워드 인자로 넘기지 않고 속성으로 직접 할당하는 것이 Isaac Lab 관례입니다.

---

## 3. USD 에셋 및 로봇 설정

### USD 파일 구조

- **경로**: `/home/yeseul/Desktop/Digital_Twin_UR/UR_Robot_System.usd`
- **아티큘레이션 루트**: `UR_Robot_System` (단일 articulation)

```
UR_Robot_System  (ArticulationRoot)
└── ur3_with_gripper
    ├── base_link            ← FrameTransformer 소스 프림
    ├── shoulder_pan_link
    │   └── shoulder_pan_joint
    ├── ... (arm joints)
    ├── wrist_3_link
    └── short_gripper        ← EE 프림 + SurfaceGripper 프림
PrismaticJoint               ← X축 테이블 슬라이딩 조인트
```

### `UR_ROBOT_SYSTEM_CFG` 주요 설정

`isaaclab_assets/robots/universal_robots.py`에 정의된 `ArticulationCfg`입니다.

| 항목 | 값 | 이유 |
|------|----|------|
| `fix_root_link=True` | 월드에 고정 | 로봇 베이스는 움직이지 않음 |
| `disable_gravity=True` | 중력 비활성화 | 실제 로봇의 중력 보상 제어기 모사 |
| `init_state.pos` | `(0.0, 0.0, 0.7356)` | 테이블 위 바닥 높이 |
| 암 초기 자세 | `[0, -π/2, π/2, -π/2, -π/2, 0]` | 중간 범위의 준비 자세 |
| 암 stiffness / damping | 800.0 / 40.0 | 임피던스 제어 느낌의 응답성 |
| 테이블 stiffness / damping | 1000.0 / 100.0 | 더 rigid한 프리즈매틱 이동 |

### SurfaceGripper 설정

Isaac Sim의 `SurfaceGripper`는 물리적인 파지를 시뮬레이션 없이 constraint 기반으로 물체를 고정합니다. CPU pipeline에서만 동작합니다.

```python
SurfaceGripperCfg(
    prim_path="{ENV_REGEX_NS}/UR_Robot_System/ur3_with_gripper/short_gripper",
    max_grip_distance=0.0075,    # 7.5mm 이내에서만 파지 시도
    shear_force_limit=5000.0,    # 수평 방향 최대 힘
    coaxial_force_limit=5000.0,  # 축 방향 최대 힘
    retry_interval=0.05,         # 파지 실패 시 재시도 간격
)
```

---

## 4. MDP 설계 원칙 및 과정

강화학습 환경을 설계할 때 MDP(Markov Decision Process)의 각 요소를 차례로 결정합니다. 아래 순서로 설계하면 요소 간 의존성을 관리하기 쉽습니다:

```
Scene → Action → Command → Observation → Reward → Termination → Event
```

### 4.1 Scene 구성

**설계 질문:** "로봇이 태스크를 수행하는 데 물리적으로 어떤 오브젝트가 필요한가?"

| 이름 | 타입 | 역할 |
|------|------|------|
| `robot` | `ArticulationCfg` | UR arm + suction gripper + prismatic table |
| `surface_gripper` | `SurfaceGripperCfg` | 흡착 그리퍼 constraint 시뮬레이션 |
| `ee_frame` | `FrameTransformerCfg` | EE 위치·회전을 world frame으로 추적 |
| `ground` | `GroundPlaneCfg` | 바닥면 |
| `light` | `DomeLightCfg` | 렌더링용 조명 |

**FrameTransformer 설계:**

```python
FrameTransformerCfg(
    prim_path="..../base_link",      # 소스: 로봇 베이스
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="..../short_gripper",
            name="end_effector",
            offset=OffsetCfg(pos=[0.15, 0.0, 0.0]),  # 그리퍼 끝단 오프셋
        )
    ],
)
```

`base_link`를 소스로 쓰는 이유: 로봇 베이스가 고정되어 있으므로 소스-타겟 변환이 곧 조인트 공간 구성에 따른 EE 위치를 반영합니다. `target_pos_w`, `target_quat_w`로 world frame 좌표를 직접 얻을 수 있습니다.

### 4.2 Action 설계

**설계 질문:** "정책(policy)이 어떤 신호를 출력해서 로봇을 제어할 것인가?"

총 action 차원: **8** (arm 6 + table 1 + gripper 1)

| 이름 | 타입 | 차원 | 비고 |
|------|------|------|------|
| `arm_action` | `RelativeJointPositionActionCfg` | 6 | 증분(delta) 제어, scale=0.0625 |
| `table_action` | `JointPositionActionCfg` | 1 | 절대 위치 제어, scale=1.0 |
| `gripper_action` | `SurfaceGripperBinaryActionCfg` | 1 | −1.0=open, +1.0=close |

**설계 결정 이유:**

- **Relative (증분) vs Absolute (절대) 제어:** 암 관절에 상대 제어를 쓴 이유는 정책 출력이 "얼마나 움직일지"를 나타내기 때문에 학습 초기 큰 관절 점프를 방지하고 부드러운 궤적이 나옵니다. 반면 테이블은 단순한 1D 슬라이딩이라 절대 위치로 직접 지정하는 편이 학습이 빠릅니다.
- **scale=0.0625:** 매 policy step(1/30 s)에 최대 ≈3.6°만 움직이도록 제한. 너무 크면 충돌 위험, 너무 작으면 학습 느림.
- **use_zero_offset=True:** 출력이 0일 때 현재 관절 위치를 유지 (정지 상태 = 0 출력).

**Isaac Lab Action 처리 흐름:**

```
정책 출력 (raw action)
    → ActionManager.process_action()    # scale 적용, clip
    → ActionManager.apply_action()      # PhysX actuator로 전달
    → decimation 횟수만큼 sim step 실행
    → 다음 observation 수집
```

### 4.3 Observation 설계

**설계 질문:** "정책이 태스크를 수행하기 위해 어떤 정보가 필요한가? 그리고 실제 로봇에서도 얻을 수 있는 정보인가?"

#### Pick-and-Place / 공통 관찰 (24차원)

| 이름 | 함수 | 차원 | 프레임 | 노이즈 |
|------|------|------|--------|--------|
| `joint_pos` | `mdp.joint_pos` | 7 | 관절 공간 | Uniform ±0.01 rad |
| `joint_vel` | `mdp.joint_vel` | 7 | 관절 공간 | Uniform ±0.01 rad/s |
| `ee_pos_w` | `mdp.ee_pos_w` | 3 | env-local (월드 − env origin) | 없음 |
| `ee_quat_w` | `mdp.ee_quat_w` | 4 | 월드 | 없음 |
| `goal_pos` | `mdp.goal_pos_command` | 3 | env-local | 없음 |
| `suction_state` | `mdp.suction_state` | 1 | scalar | 없음 |

#### Align 태스크 추가 관찰

| 이름 | 함수 | 차원 | 설명 |
|------|------|------|------|
| `goal_quat` | `mdp.goal_quat_command` | 4 | 목표 회전(quaternion) |

**설계 결정 이유:**

- **env-local frame 사용:** `ee_pos_w = world_pos − env_origins`로 환경 원점 기준 좌표를 씁니다. 멀티-env에서 각 환경의 절대 world 좌표는 다르지만, env-local 좌표는 동일한 의미를 가집니다. 정책이 더 빠르게 패턴을 학습합니다.
- **joint_pos + joint_vel:** 로봇 상태를 완전히 나타내기 위한 최소 정보. 실제 로봇에서 엔코더로 측정 가능.
- **ee_pos + ee_quat:** forward kinematics를 정책이 내부에서 학습할 필요 없이 직접 제공. 수렴 속도 향상.
- **goal_pos (+ goal_quat):** 정책이 "어디로 가야 하는지" 알아야 방향성 있는 행동 가능. 없으면 reward signal만으로 목표를 추측해야 하므로 학습이 매우 느려짐.
- **노이즈 (Uniform ±0.01):** 실제 로봇의 엔코더 측정 오차를 모사. sim-to-real gap 감소에 기여.

**`enable_corruption`:** 학습 시 `True`(노이즈 적용), PLAY 시 `False`(클린 관찰).

#### `goal_pos_command` 프레임 일관성

```
UniformPoseCommand → command[:, :3]
  = robot base frame 기준 목표 위치
  ≈ env-local frame (robot base가 env origin에 고정되어 있으므로)

ee_pos_w = target_pos_w - env_origins
  = env-local frame

→ 두 값이 같은 프레임 → 직접 차이 계산 가능
```

### 4.4 Command 설계

**설계 질문:** "에피소드마다 어떤 목표를 주어야 policy가 일반화된 능력을 학습하는가?"

```python
UniformPoseCommandCfg(
    asset_name="robot",
    body_name="short_gripper",       # command가 이 link 기준으로 샘플링됨
    resampling_time_range=(8.0, 8.0),
    ranges=UniformPoseCommandCfg.Ranges(
        pos_x=(0.3, 0.7),    # 테이블 X 방향 작업 공간
        pos_y=(-0.25, 0.25), # 테이블 Y 방향 작업 공간
        pos_z=(0.06, 0.06),  # 테이블 표면 높이 고정
        roll=(3.14, 3.14),   # EE 아래 방향 고정 (흡착을 위해)
        pitch=(0.0, 0.0),
        yaw=(-3.14, 3.14),   # 회전은 자유롭게
    ),
)
```

**설계 결정 이유:**

- **`resampling_time_range=(8.0, 8.0)`:** 에피소드 중간에 목표가 바뀌지 않도록 에피소드 길이보다 길게 설정. 한 에피소드 = 하나의 고정 목표.
- **pos_z 고정:** 흡착 그리퍼는 수직 방향 접근이 필요. 높이를 고정하면 학습이 훨씬 단순해짐.
- **yaw 랜덤:** 테이블 위 물체가 다양한 방향으로 놓일 수 있으므로 EE orientation을 다양하게 학습시킴.
- **X 범위 0.3~0.7:** 로봇 베이스에서 팔 길이 고려, 충돌 없는 도달 가능 범위.

### 4.5 Reward 설계

**설계 원칙:** Reward는 정책이 원하는 행동을 배울 수 있도록 충분한 신호를 주되, 의도하지 않은 행동(reward hacking)을 유발하지 않아야 합니다.

#### Align 태스크 Reward

| 이름 | 함수 | 가중치 | 수식 | 역할 |
|------|------|--------|------|------|
| `ee_pos_align` | `ee_goal_pos_reward` | +1.0 | `exp(-10 × dist(ee, goal))` | 위치 정렬 |
| `ee_ori_align` | `ee_goal_ori_reward` | +0.2 | `|q_ee · q_goal|` | 방향 정렬 |
| `action_rate` | `action_rate_l2` | −0.01 | `‖a_t − a_{t-1}‖²` | 부드러운 동작 |
| `action` | `action_l2` | −0.01 | `‖a_t‖²` | 에너지 절약 |

**`ee_goal_pos_reward` 설계:**

```python
dist = ||ee_pos_env_local - goal_pos_env_local||₂
reward = exp(-alpha * dist)
```

- **Exponential 형태:** 거리가 0에 가까울수록 보상이 1에 수렴하는 형태. 멀리 있을 때도 작은 gradient를 제공해 학습 초기 방향성을 유지함.
- **`alpha=10.0`:** 10cm 거리에서 보상 ≈ 0.37. 3cm에서 ≈ 0.74. 너무 크면 목표 근처에서만 학습이 일어나고, 너무 작으면 어디서나 비슷한 보상.
- **Sparse보다 Dense 선호:** 흡착 로봇 태스크에서 sparse reward(성공 시에만 보상)만 사용하면 탐색이 너무 어려워 수렴이 매우 느림. Dense reward로 방향 신호를 계속 제공.

**`ee_goal_ori_reward` 설계:**

```python
reward = |q_ee · q_goal|  # 절댓값으로 q, -q 동일성 처리
```

- **Quaternion dot product:** 두 회전이 동일하면 `|q·q'| = 1`, 90° 차이면 ≈ 0. 직관적이고 계산 효율적.
- **낮은 가중치(0.2):** 위치 정렬이 더 중요하므로 방향 reward는 보조적으로만 사용. 너무 크면 방향만 맞추고 위치를 무시하는 local optima 발생.

**정규화 페널티:**

- `action_rate_l2`: 연속된 두 action의 차이에 L2 패널티 → 진동 억제, 부드러운 궤적
- `action_l2`: action 크기 자체에 패널티 → 불필요한 움직임 억제

**가중치 밸런싱 원칙:**

```
태스크 reward의 최대값이 정규화 패널티보다 충분히 커야 합니다.
  ee_pos_align max = 1.0
  ee_ori_align max = 0.2
  action_rate max penalty ≈ -0.01 × (max_action_change²)
                          ≈ -0.01 × (8 joints × scale²)
→ 태스크 reward가 지배적 → 정책이 태스크를 우선함
```

#### Pick-and-Place 태스크 Reward (참고용, 박스 추가 시 활성화)

```
Phase 1 (approach):  exp(-10 × dist(ee, box))       [suction OFF 때만]
Phase 2 (grasp):     1 if suction ON and dist < 5cm
Phase 3 (lift):      clamp(box_z - table_h, 0, 0.15) / 0.15   [suction ON 때만]
Phase 4 (place):     exp(-10 × dist_xy(box, goal))  [lifted 때만]
Phase 5 (success):   +10.0 (sparse bonus)
```

Phased reward의 장점: 단계별로 명확한 서브골을 제공해 긴 horizon 태스크를 분해합니다.

### 4.6 Termination 설계

**설계 질문:** "언제 에피소드를 끝내야 하는가?"

#### Align 태스크

| 이름 | 함수 | 조건 | 타입 |
|------|------|------|------|
| `time_out` | `mdp.time_out` | 8초 경과 | 자연 종료 (time_out=True) |
| `ee_success` | `mdp.ee_at_goal` | EE가 goal 3cm 이내 | 성공 종료 |

**`time_out=True`의 의미:** Isaac Lab이 PPO 학습 시 timeout으로 인한 종료에서는 value bootstrapping을 적용합니다. 일반 종료(실패 등)와 구분해서 학습 신호를 다르게 처리합니다.

**성공 termination을 두는 이유:** 목표 달성 시 에피소드를 즉시 종료함으로써:
1. 더 많은 에피소드 = 더 많은 학습 기회
2. 정책이 빠른 성공에 내재적 인센티브를 가짐
3. 성공 후 불필요한 행동으로 보상을 낭비하는 것을 방지

**threshold=0.03m (3cm):** 실제 흡착 그리퍼의 `max_grip_distance=0.0075m` 보다 여유를 두어, EE가 "충분히 가까운" 위치에 있으면 성공으로 판정.

### 4.7 Event (Reset) 설계

**설계 질문:** "에피소드 리셋 시 어떤 초기 조건의 다양성을 줘야 하는가?"

```python
reset_robot_joints = EventTerm(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
        "position_range": (-0.1, 0.1),   # ±0.1 rad ≈ ±5.7°
        "velocity_range": (0.0, 0.0),    # 초기 속도 0
    },
)
```

**설계 결정 이유:**

- **작은 랜덤 오프셋(±0.1 rad):** 너무 크면 초기 자세가 이상해 학습 초기 불안정. 너무 작으면 정책이 특정 초기 자세에만 익숙해짐. 실제 로봇 배포 시 초기 자세 오차 범위 내.
- **초기 속도 0:** 로봇이 정지 상태에서 시작. 실제 배포와 일치.

---

## 5. 태스크별 환경 설정

### 5.1 Align 태스크 (경량, 권장 시작점)

**목적:** 물체 없이 EE를 목표 포즈로 이동시키는 reach/align 태스크. Pick-and-Place 전 단계로 팔의 위치 제어 능력을 먼저 학습합니다.

```
베이스: URRobotSystemEnvCfg
오버라이드:
  - episode_length_s: 15s → 8s
  - resampling_time_range: 8s → 6s
  - observations: goal_quat 추가 (4차원)
  - rewards: ee_pos_align(+1.0) + ee_ori_align(+0.2) 추가
  - terminations: ee_success 추가 (3cm 이내)
```

**등록된 Gym ID:**

| ID | 설정 클래스 | 용도 |
|----|------------|------|
| `Isaac-URTable-Align-v0` | `URTableAlignEnvCfg` | 학습 (256 envs) |
| `Isaac-URTable-Align-Play-v0` | `URTableAlignEnvCfg_PLAY` | 평가 (4 envs, noise 없음) |

**Observation 차원 (Align):** 28차원

```
joint_pos   (7)
joint_vel   (7)
ee_pos_w    (3)
ee_quat_w   (4)
goal_pos    (3)
goal_quat   (4)    ← Align 태스크에서 추가
suction_state (1)  ← gripper 있지만 보조적
```

### 5.2 Pick-and-Place 태스크

**목적:** 테이블 위 물체를 흡착해 목표 위치로 옮기는 전체 pick-and-place 태스크.

**등록된 Gym ID:**

| ID | 설정 클래스 | 용도 |
|----|------------|------|
| `Isaac-URTable-PickPlace-v0` | `URRobotSystemEnvCfg` | 학습 |
| `Isaac-URTable-PickPlace-Play-v0` | `URRobotSystemEnvCfg_PLAY` | 평가 |

> 현재 상태: 박스를 scene에 추가하지 않아 정규화 reward만 활성화. TODO 섹션 참조.

**시뮬레이션 공통 파라미터:**

| 파라미터 | 값 | 의미 |
|----------|----|------|
| `sim.dt` | `1/120` s | 물리 시뮬레이션 스텝 크기 |
| `decimation` | 4 | 정책 스텝 = 4 × sim.dt = 1/30 s |
| `device` | `cpu` | SurfaceGripper가 CPU pipeline 필요 |
| `env_spacing` | 3.0 m | 멀티-env 간격 |

---

## 6. RSL-RL PPO 학습 설정

**설정 클래스**: `URTablePPORunnerCfg` (`agents/rsl_rl_ppo_cfg.py`)

RSL-RL은 Isaac Lab 공식 지원 RL 라이브러리입니다. `PPORunnerCfg`와 `PPOCfg`로 구성됩니다.

### 학습 루프 파라미터

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `num_steps_per_env` | 512 | 각 env가 rollout 동안 수집하는 transition 수 |
| `max_iterations` | 5000 | 최대 PPO 업데이트 횟수 |
| `save_interval` | 200 | 체크포인트 저장 주기 |
| `empirical_normalization` | True | 관찰값 running mean/std 정규화 (수렴 안정화) |

**데이터 수집량:** `num_envs × num_steps_per_env = 256 × 512 = 131,072 transitions/iteration`

### 네트워크 구조

Actor와 Critic이 별도의 MLP를 가집니다.

| 파라미터 | 값 |
|----------|----|
| hidden dims | [256, 256, 128] |
| activation | ELU |
| `init_noise_std` | 1.0 (초기 탐색 범위) |

**[256, 256, 128]을 선택한 이유:** Observation 차원(28)이 작으므로 과도하게 큰 네트워크는 불필요. 3-layer MLP로 충분한 표현력 확보.

### PPO 알고리즘 파라미터

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `learning_rate` | 3e-4 | adaptive KL 기반으로 자동 조정 |
| `desired_kl` | 0.008 | 목표 KL divergence (너무 크면 정책 급변) |
| `clip_param` | 0.2 | PPO clipping (표준값) |
| `num_learning_epochs` | 8 | 하나의 rollout 데이터로 몇 번 업데이트 |
| `num_mini_batches` | 8 | mini-batch 수 (배치 크기 = 131072/8 = 16384) |
| `gamma` | 0.99 | 할인율 (먼 미래도 고려) |
| `lam` | 0.95 | GAE lambda (bias-variance tradeoff) |
| `entropy_coef` | 0.005 | 엔트로피 보너스 (탐색 장려) |
| `max_grad_norm` | 1.0 | gradient clipping |

---

## 7. 실행 방법

### Random Agent 테스트 (환경 동작 확인)

```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
  --task Isaac-URTable-Align-v0 --num_envs 4
```

### RSL-RL 학습 (Align 태스크)

```bash
# 헤드리스 학습 (권장, 속도 빠름)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-URTable-Align-v0 --headless

# 렌더링하며 학습 (초기 디버깅 시)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-URTable-Align-v0 --num_envs 16
```

### 학습 결과 평가

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-URTable-Align-Play-v0 \
  --checkpoint logs/rsl_rl/Isaac-URTable-Align-v0/<run>/model_<iter>.pt \
  --num_envs 1
```

### TensorBoard 모니터링

```bash
tensorboard --logdir logs/rsl_rl/Isaac-URTable-Align-v0
```

주요 모니터링 지표:
- `Train/mean_reward`: 평균 에피소드 보상 (올라가야 정상)
- `Train/mean_episode_length`: 평균 에피소드 길이 (성공 termination 이면 짧아짐)
- `Train/ee_pos_align`: 위치 정렬 reward 기여도

---

## 8. 코드 흐름 추적

### 환경 등록 → 생성 흐름

```
__init__.py
  gym.register("Isaac-URTable-Align-v0", ...)
      ↓
isaaclab.sh -p train.py --task Isaac-URTable-Align-v0
      ↓
gym.make("Isaac-URTable-Align-v0")
      ↓
ManagerBasedRLEnv(env_cfg=URTableAlignEnvCfg())
      ↓
URTableAlignEnvCfg.__post_init__()
  → super().__post_init__()  (URRobotSystemEnvCfg)
    → super().__post_init__()  (URTableEnvCfg)
  → rewards에 ee_pos_align, ee_ori_align 추가
  → terminations에 ee_success 추가
```

### 학습 루프의 1 iteration

```
RSL-RL Runner
  ↓
for step in range(num_steps_per_env):
    obs = env.get_observations()          # ObservationManager 수집
    action = policy(obs)                  # Actor 네트워크 추론
    obs, reward, done, info = env.step(action)
        → ActionManager.process_action()  # scale, clip
        → for _ in range(decimation):
              sim.step()                  # PhysX 4 스텝
        → ObservationManager.compute()   # 새 관찰 수집
        → RewardManager.compute()        # 각 reward term 합산
        → TerminationManager.compute()   # done 판정
        → EventManager (reset if done)
    rollout_buffer.add(obs, action, reward, done)
  ↓
PPO.update(rollout_buffer)               # 8 epochs × 8 mini-batches
```

### Reward 계산 흐름

```python
# RewardManager가 매 policy step마다 호출
total_reward = 0
for term_name, term_cfg in rewards_cfg.items():
    term_value = term_cfg.func(env, **term_cfg.params)  # (num_envs,)
    total_reward += term_cfg.weight * term_value

# ee_goal_pos_reward 내부:
ee_frame = env.scene["ee_frame"]
ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins
goal_pos = env.command_manager.get_command("goal_pose")[:, :3]
dist = torch.norm(ee_pos - goal_pos, dim=-1)
return torch.exp(-10.0 * dist)  # (num_envs,)
```

---

## 9. TODO

### 단기

- [ ] Align 태스크 학습 후 수렴 여부 확인 및 하이퍼파라미터 튜닝 (`alpha`, reward 가중치)
- [ ] 학습된 Align 정책으로 EE → goal 성공률 측정

### 중기 (Pick-and-Place 완성)

- [ ] box를 Scene에 추가 (`RigidObjectCfg`)하고 테이블 표면 높이 확인
- [ ] box 관련 observation 추가 (`box_pos_w`, `box_quat_w`)
- [ ] phased pick-and-place reward 활성화 (`approach_box`, `grasp`, `lift_box`, `place_box`, `success_bonus`)
- [ ] box 관련 termination 추가 (`box_at_goal`, `box_dropped`)
- [ ] Align 정책을 초기 정책으로 활용한 curriculum learning 검토

### 장기

- [ ] GPU pipeline 전환 가능 여부 검토 (SurfaceGripper CPU 제약 해결 시)
- [ ] ROS Inference 변형 환경 추가 (실제 로봇 배포용)
- [ ] sim-to-real gap 줄이기: observation noise 파라미터 실제 엔코더 스펙 기준으로 교정
