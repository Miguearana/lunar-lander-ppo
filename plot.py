import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load(r"C:\Users\arana\lunar_lander_ppo")
env = gym.make("LunarLander-v3")

rewards = []
for ep in range(20):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    rewards.append(total_reward)
    print(f"Episode {ep+1}: {total_reward:.2f}")

env.close()

plt.figure(figsize=(10, 5))
plt.plot(rewards, marker='o', color='royalblue', linewidth=2)
plt.axhline(y=200, color='green', linestyle='--', label='Solved threshold (200)')
plt.axhline(y=sum(rewards)/len(rewards), color='orange', linestyle='--', label=f'Mean: {sum(rewards)/len(rewards):.1f}')
plt.title("PPO Agent — LunarLander-v3 Evaluation (20 episodes)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\ml\rewards_plot.png", dpi=150)
plt.show()
print("Plot saved to C:\\ml\\rewards_plot.png")
