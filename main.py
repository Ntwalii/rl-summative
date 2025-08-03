from environment.custom_env import FlowEnv
from environment.rendering_flow import draw_message
from stable_baselines3 import DQN, PPO
import os

def run_agent(model, agent_name="Agent", episodes=3, max_steps=50, record=False):
    env = FlowEnv()
    frame_capture = record  # Only record first agent's first episode
    print(f"\n🔁 Running {agent_name} for {episodes} episodes")

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        print(f"\n🎬 {agent_name} - Episode {ep + 1}")
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            draw_message(obs, action, step, record=frame_capture)
            total_reward += reward
            if done:
                break
        print(f"✅ {agent_name} Episode {ep + 1} total reward: {total_reward}")
        frame_capture = False  # Only record first episode

    env.close()

def main():
    # Optional: clean gif_frames/ before recording
    if os.path.exists("gif_frames"):
        for f in os.listdir("gif_frames"):
            os.remove(os.path.join("gif_frames", f))
    else:
        os.makedirs("gif_frames", exist_ok=True)

    # Load models
    dqn_model = DQN.load("models/flow/flow_dqn_agent")
    ppo_model = PPO.load("models/flow/flow_ppo_agent")

    # Run both agents, record DQN agent only
    run_agent(dqn_model, agent_name="DQN Agent", record=True)
    run_agent(ppo_model, agent_name="PPO Agent", record=False)

if __name__ == "__main__":
    main()
