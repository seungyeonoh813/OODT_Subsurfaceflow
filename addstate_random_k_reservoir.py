import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matlab.engine
import os

class ReservoirEnv(gym.Env):
    eng = None 

    def __init__(self):                   #Delete k for the input
        super(ReservoirEnv, self).__init__()
        
        self.s_max = 170.0               
        self.hard_limit = 200.0          
        self.q_max_m3_per_day = 200.0    

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space: 1024 (32x32 Grid ) + 1(random parameter k) = 1025
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1025,), dtype=np.float32)

        if ReservoirEnv.eng is None:
            print("MATLAB engine connecting...")
            ReservoirEnv.eng = matlab.engine.start_matlab()
            ReservoirEnv.eng.addpath(os.getcwd(), nargout=0)
        
        self.dt = 1.0          
        self.T = 30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        
        self.init_p = np.random.uniform(95.0, 105.0)
        
        # logk = 1 + (2-1)*rand() -> K = 10 ~ 100 mD (random k)
        log_k = np.random.uniform(1.0, 2.0)
        self.k_mD = 10.0 ** log_k  
        
        res = ReservoirEnv.eng.random_initial_k(
            float(0.0), float(self.dt), True, float(self.init_p), float(self.k_mD), nargout=1
        )
        
        # To make the observation space normalized, previously did the normalization directly but we have k, will do seperately
        p_array = np.array(res['pressure']).flatten() / 1e5
        p_norm = ((p_array - 100.0) / 100.0).astype(np.float32)
        
        # k normalization
        k_norm = np.array([self.k_mD / 100.0], dtype=np.float32)
        
        # Combine the pressure (grid) and k (normalized)
        obs = np.concatenate((p_norm, k_norm))
        
        return obs, {'init_p': self.init_p, 'k_mD': self.k_mD}

    def step(self, action):
        u_clip = np.clip(float(action[0]), -1.0, 1.0)
        u_norm = (u_clip + 1.0) / 2.0
        q_inj_day = 100.0 + u_norm * (self.q_max_m3_per_day - 100.0) 
        # q_inj_day = 0.0 + u_norm * self.q_max_m3_per_day
        
        res = ReservoirEnv.eng.random_initial_k(
            float(q_inj_day), float(self.dt), False, 0.0, 0.0, nargout=1
        )
        
        p_array = np.array(res['pressure']).flatten() / 1e5
        max_p = float(np.max(p_array))
        self.steps_taken += 1

        r_control = u_norm * 30.0
        r_penalty = 0.0
        r_terminal = 0.0
        terminated = False

        if self.steps_taken >= self.T:
            terminated = True

        if max_p > self.hard_limit:
            terminated = True         
            r_terminal = -50.0        
            r_penalty = -2.0 * (self.hard_limit - self.s_max) 
        else:
            if max_p > self.s_max:
                r_penalty = -2.0 * (max_p - self.s_max)

        total_reward = r_control + r_penalty + r_terminal
        
        # Observation space ( grid + k )
        p_norm = ((p_array - 100.0) / 100.0).astype(np.float32)
        k_norm = np.array([self.k_mD / 100.0], dtype=np.float32)
        obs = np.concatenate((p_norm, k_norm))
        
        info = {
            "max_pressure": max_p,
            "u": q_inj_day,
            "total_reward": total_reward,
            "k_mD": self.k_mD
        }
        
        return obs, float(total_reward), terminated, False, info

    def close(self):
        pass