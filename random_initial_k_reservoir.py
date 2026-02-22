import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matlab.engine
import os

class ReservoirEnv(gym.Env):
    eng = None 

    def __init__(self, k_mD):
        super(ReservoirEnv, self).__init__()

        self.k_mD = k_mD
        
        # FDM 코드와 동일한 물리적 제약 조건
        self.s_max = 170.0               # Soft Limit (Bar)
        self.hard_limit = 200.0          # Hard Limit (Bar)
        self.q_max_m3_per_day = 200.0    # 최대 주입량

        # Action Space: [0.0, 1.0] (회원님 설계 원본)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space: FDM과 동일하게 전체 Grid (32x32 = 1024) 압력 정보 제공
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32)

        if ReservoirEnv.eng is None:
            print("MATLAB engine connecting...")
            ReservoirEnv.eng = matlab.engine.start_matlab()
            ReservoirEnv.eng.addpath(os.getcwd(), nargout=0)
        
        # FDM 코드와 동일하게 30스텝 종료 세팅 (1스텝 = 1일, 총 30일)
        self.dt = 1.0          
        self.T = 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        
        # 초기 압력 (95 ~ 105 Bar) - FDM 코드 로직
        self.init_p = np.random.uniform(95.0, 105.0)
        
        res = ReservoirEnv.eng.random_initial_k(
            float(0.0), float(self.dt), True, float(self.init_p), float(self.k_mD), nargout=1
        )
        
        p_array = np.array(res['pressure']).flatten() / 1e5
        
        # Observation 정규화: (p - 100) / 100
        obs = ((p_array - 100.0) / 100.0).astype(np.float32)
        return obs, {'init_p': self.init_p}

    def step(self, action):
        u_clip = np.clip(float(action[0]), -1.0, 1.0)
        u_norm = (u_clip + 1.0) / 2.0
        
        # FDM 코드와 완벽히 동일한 주입량 매핑 (100 ~ 200 사이 탐색)
        q_inj_day = 100.0 + u_norm * (self.q_max_m3_per_day - 100.0)
        
        # MATLAB 시뮬레이터 1스텝 전진
        res = ReservoirEnv.eng.random_initial_k(
            float(q_inj_day), float(self.dt), False, 0.0, 0.0, nargout=1
        )
        
        p_array = np.array(res['pressure']).flatten() / 1e5
        max_p = float(np.max(p_array))
        
        self.steps_taken += 1

        # -----------------------------------------------------------
        # 회원님의 FDM 보상/페널티 논리 100% 적용 (Linear Reward)
        # -----------------------------------------------------------
        r_control = u_norm * 30.0
        r_penalty = 0.0
        r_terminal = 0.0
        terminated = False

        if self.steps_taken >= self.T:
            terminated = True

        # 파괴 조건 (Hard Constraint > 200 Bar) 우선 확인
        if max_p > self.hard_limit:
            terminated = True         # 즉시 종료
            r_terminal = -50.0        # 파괴 페널티 -50 때리기
            
            # FDM 코드처럼 200 Bar에서 멈춘 상황의 물리적 페널티만 부과합니다.
            # (MRST가 비현실적으로 800 Bar까지 계산해서 -1300이 되는 것을 방지)
            r_penalty = -2.0 * (self.hard_limit - self.s_max) 
        else:
            # 정상 생존 시 (Soft Constraint > 170 Bar)
            if max_p > self.s_max:
                r_penalty = -2.0 * (max_p - self.s_max)

        total_reward = r_control + r_penalty + r_terminal

        # Observation 정규화
        obs = ((p_array - 100.0) / 100.0).astype(np.float32)
        
        info = {
            "max_pressure": max_p,
            "u": q_inj_day,
            "r_control": r_control,
            "r_penalty": r_penalty,
            "r_terminal": r_terminal,
            "total_reward": total_reward
        }
        
        return obs, float(total_reward), terminated, False, info

    def close(self):
        pass