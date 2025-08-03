# AI-Powered Scam Message Filter (Reinforcement Learning Project)

This project applies reinforcement learning (RL) to a simulated message moderation system, where an agent learns to identify and handle scam messages flowing through a digital feed. The agent is trained in a custom-built non-grid environment (`FlowEnv`) and learns to take real-time moderation actions like **allow**, **block**, or **investigate**.

---

## Objective

Design a reinforcement learning agent that:
- Maximizes reward by correctly identifying scams
- Minimizes false positives (blocking safe content)
- Handles messages with varying urgency and sender reputation

---

## Environment: `FlowEnv`

Each message is represented by:
- `sender_reputation` ∈ [0.0, 1.0]
- `urgency_score` ∈ [0.0, 1.0]
- `msg_type` ∈ {0 = good, 1 = suspicious, 2 = scam}

### Agent Actions:
- `0`: Allow  
- `1`: Block  
- `2`: Investigate

### Reward Structure:
| Action      | Good | Suspicious | Scam |
|-------------|------|------------|------|
| Allow       | 0    | 0          | -5   |
| Block       | -10  | -10        | +10  |
| Investigate | -2   | +5         | +2   |

---

## Algorithms Implemented

- **Deep Q-Network (DQN)** using experience replay, target network, ε-greedy exploration
- **Proximal Policy Optimization (PPO)** with entropy regularization and GAE

---

## Visualization

The `pygame` renderer (`rendering_flow.py`) shows:
- Falling message blocks (color-coded by risk)
- Agent decisions displayed live
- Optional GIF recording using `imageio`


