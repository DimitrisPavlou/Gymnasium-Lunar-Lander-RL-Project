## 🚀 Project Overview

This project focuses on **Reinforcement Learning (RL)** and provides implementations of several fundamental RL agents to solve both the **discrete** and **continuous** versions of the **Lunar Lander environment** from the [Gymnasium](https://gymnasium.farama.org/) library by OpenAI.  

---

## 🧠 Implemented RL Agents

The following agents are implemented and tested on the Lunar Lander environments:

- **Value-based methods**
  - Deep Q-Network (**DQN**)  
  - Double Deep Q-Network (**DDQN**)  
  - Dueling DQN  
  - Dueling DDQN  

- **Policy-based & Actor-Critic methods**
  - Advantage Actor-Critic (**A2C**)  
  - Proximal Policy Optimization (**PPO**)  
  - Deep Deterministic Policy Gradient (**DDPG**)  

---

## 📦 Memory Buffers

Two types of replay buffers are included:  

1. **Uniform Replay Buffer** – traditional random sampling of experiences  
2. **Prioritized Experience Replay (PER)** – samples experiences with higher learning potential more frequently  

---

## 🌳 Rule Extraction Algorithm

A key addition to this project is a **Rule Extraction module** that bridges reinforcement learning and interpretable machine learning:  

1. A pre-trained RL agent is run in the environment to **collect experiences**.  
2. These experiences are used to **build a dataset** of state-action mappings.  
3. A simple supervised model (e.g., **Decision Tree**, or optionally a small MLP) is trained on this dataset.  
4. The resulting model achieves **similar performance** to the original RL agent while being more **interpretable**.  

Currently, rule extraction is demonstrated for the **discrete Lunar Lander**, but the method can be generalized to continuous action spaces.  

---

## ✨ Key Features

- Supports both **discrete** and **continuous** environments  
- Multiple **RL algorithms** in a unified framework  
- Flexible **experience replay strategies**  
- Integration of **rule-based interpretability** into RL workflows  
