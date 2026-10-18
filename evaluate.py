import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load(r"C:\Users\arana\lunar_lander_ppo")

env = gym.make("LunarLander-v3", render_mode="human")

episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep+1}: reward = {total_reward:.2f}")

env.close()
