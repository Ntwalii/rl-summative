import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class FlowEnv(gym.Env):
    def __init__(self):
        super(FlowEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Allow, 1: Block, 2: Investigate

        # Observation: [sender_reputation (0–1), urgency (0–1), msg_type (0: good, 1: suspicious, 2: scam)]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.current_message = self._generate_message()
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_message = self._generate_message()
        self.step_count = 0
        return self.current_message.copy(), {}

    def step(self, action):
        msg_type = int(self.current_message[2])
        reward = 0

        if action == 0:  # Allow
            if msg_type == 2:
                reward = -5  # scam allowed
            else:
                reward = 0  # safe or suspicious allowed
        elif action == 1:  # Block
            if msg_type == 2:
                reward = +10  # blocked scam!
            else:
                reward = -10  # blocked safe/suspicious
        elif action == 2:  # Investigate
            if msg_type == 1:
                reward = +5
            elif msg_type == 2:
                reward = +2  # scam under investigation
            else:
                reward = -2  # waste of time

        self.current_message = self._generate_message()
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.current_message.copy(), reward, done, False, {}

    def _generate_message(self):
        sender_reputation = np.clip(np.random.normal(0.5, 0.2), 0, 1)
        urgency = np.clip(np.random.beta(2, 5), 0, 1)
        msg_type = np.random.choice([0, 1, 2], p=[0.6, 0.25, 0.15])  # mostly good
        return np.array([sender_reputation, urgency, float(msg_type)], dtype=np.float32)

    def render(self):
        print(f"Message: sender={self.current_message[0]:.2f}, urgency={self.current_message[1]:.2f}, type={int(self.current_message[2])}")

    def close(self):
        pass
