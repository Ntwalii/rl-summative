from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import FlowEnv
import os
import time

def train_agent(timesteps=10000):
    env = DummyVecEnv([FlowEnv])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        gamma=0.99,
        batch_size=64,
        n_steps=128,
        ent_coef=0.01,
        gae_lambda=0.95,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
    )

    print("🚀 Starting PPO training...")
    start = time.time()
    model.learn(total_timesteps=timesteps)
    duration = time.time() - start
    print(f"✅ Training completed in {duration:.2f}s")

    # Save model
    os.makedirs("models/flow", exist_ok=True)
    model.save("models/flow/flow_ppo_agent")
    print("💾 PPO model saved to models/flow/flow_ppo_agent.zip")

    return model

def evaluate(model, episodes=5):
    env = FlowEnv()
    total_reward = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
        print(f"Episode {ep+1}: Reward = {ep_reward}")
    avg = total_reward / episodes
    print(f"\n📈 PPO Average Reward over {episodes} episodes: {avg:.2f}")

if __name__ == "__main__":
    model = train_agent(timesteps=500_000)
    evaluate(model)
