import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from addstate_random_k_reservoir import ReservoirEnv

class TrainingCurveCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.step_rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        self.step_rewards.append(self.locals['rewards'][0])
        self.timesteps.append(self.num_timesteps)
        return True

# Delete k
env = ReservoirEnv() 

callback = TrainingCurveCallback()
model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, ent_coef=0.01)

print("Training Start")
model.learn(total_timesteps=20000, callback=callback)
print("Finished!")

n_test = 5
all_u, all_p = [], []
test_k_values = []   # to store k (for each episode)

print(f"\nTesting {n_test} episodes with Random K...")

for i in range(n_test):
    obs, info = env.reset()
    current_k = info['k_mD']
    test_k_values.append(current_k)
    
    print(f"\n--- Episode {i+1} [Permeability (K) = {current_k:.1f} mD] ---")
    done = False
    temp_u, temp_p = [], []
    step_count = 1

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        temp_u.append(info['u'])
        temp_p.append(info['max_pressure'])

        if step_count % 5 == 0 or done:
            print(f"Step {step_count:2d} | u: {info['u']:.1f} m3/d | Max P: {info['max_pressure']:.1f} bar | "
                  f"Total Reward: {info['total_reward']:.3f}")
        
        step_count += 1

    while len(temp_u) < env.T:
        temp_u.append(temp_u[-1])
        temp_p.append(temp_p[-1])
        
    all_u.append(temp_u)
    all_p.append(temp_p)

arr_u = np.array(all_u)
arr_p = np.array(all_p)


# ==========================================
# Visualize
# ==========================================
print("Plotting Integrated Dashboard...")
plt.figure(figsize=(10, 12))
time_axis = np.arange(1, env.T + 1) * env.dt

# [1. Training curve]
ax1 = plt.subplot(3, 1, 1)
ax1.plot(callback.timesteps, callback.step_rewards, label='Step Reward', color='blue', alpha=0.2)
window_size = 300
if len(callback.step_rewards) > window_size:
    avg = np.convolve(callback.step_rewards, np.ones(window_size)/window_size, mode='valid')
    ax1.plot(callback.timesteps[window_size-1:], avg, color='red', linewidth=2, label=f'Avg ({window_size} steps)')
ax1.set_title('1. Training Curve')
ax1.set_ylabel('1-Step Reward')
ax1.legend(); ax1.grid(True, alpha=0.3)

# [2. Injection]
ax2 = plt.subplot(3, 1, 2)
for i in range(n_test): 
    ax2.plot(time_axis, arr_u[i], '-', alpha=0.4, label=f'Ep {i+1} (k={test_k_values[i]:.0f})' if i < 5 else "")
ax2.set_title('2. Injection per Permeability (k)')
ax2.set_ylabel('Injection Rate (m3/day)')
ax2.set_ylim(0, 210)
ax2.legend(loc='lower right', fontsize='small'); ax2.grid(True, alpha=0.3)

# [3. Max pressure]
ax3 = plt.subplot(3, 1, 3, sharex=ax2)
for i in range(n_test): 
    ax3.plot(time_axis, arr_p[i], '-', alpha=0.4)
ax3.axhline(env.s_max, color='k', linestyle='--', label='s_max (170 bar)')
ax3.axhline(env.hard_limit, color='r', linestyle=':', label='Hard Limit (200 bar)')
ax3.set_title('3. Max Pressure')
ax3.set_xlabel('Time (Days)')
ax3.set_ylabel('Max Pressure (Bar)')
ax3.set_ylim(100, 200)
ax3.legend(loc='lower right'); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()