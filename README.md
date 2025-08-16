# Gymnasium-Lunar-Lander-RL-Project
This is a reinforcement learning project, that solves the discrete and continuous Lunar Lander Environment from the Gymnasium library from OpenAI. 

I have implemented all the basic RL Agents that work for both the discrete and continuous version of the environment. More specifically I implemented the following: 
i) DQN and DDQN 
ii) Dueling DQN and Dueling DDQN
iii) A2C 
iv) PPO
v) DDPG

There are also two types of memory buffers: 
i) The traditional random buffer 
ii) Prioritized Experience Replay

The final addition to the project is a Rule Extaction Algorithm. The main goal is to use a trained agent and run the environent on the trained agent. While running the environment we collect experiences and create a dataset from them. We then use this dataset to train a simple machine learning model like a Decision Tree (this can be another more complex model like a simple MLP) and show that the decision tree when trained with the extracted rules achieves similar performance with the pre-trained agent. The tree is used only in the discrete environment case but this can be generalized for the continuous version. 
