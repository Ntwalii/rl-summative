from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import FlowEnv
import os
import time

def train_agent(timesteps):
    env = DummyVecEnv([FlowEnv])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        gamma=0.99,
        batch_size=64,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        buffer_size=10000,
        learning_starts=500,
        train_freq=1,
        target_update_interval=250,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
    )

    print("🚀 Starting training...")
    start = time.time()
    model.learn(total_timesteps=timesteps)
    duration = time.time() - start
    print(f"✅ Training completed in {duration:.2f}s")

    # Save model
    os.makedirs("models/flow", exist_ok=True)
    model.save("models/flow/flow_dqn_agent")
    print("💾 Model saved to models/flow/flow_dqn_agent.zip")

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
    print(f"\n📈 Average Reward over {episodes} episodes: {avg:.2f}")

if __name__ == "__main__":
    model = train_agent(timesteps=500_000)
    evaluate(model)
