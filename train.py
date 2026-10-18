import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("LunarLander-v3")

model = PPO("MlpPolicy", env, verbose=1)
print("Training started...")
model.learn(total_timesteps=500_000)
model.save("lunar_lander_ppo")
print("Model saved.")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
