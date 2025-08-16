# @title Neccessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from models import ActorDDPG, CriticDDPG
from Buffers import OUNoise, ReplayBuffer
# @title DDPG
#build ddpg agent


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 5e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic


class DDPGAgent:

    def __init__(self , state_size, action_size, device,random_seed, add_noise = False):
        #INPUT:
        #state_size: dimension of the observation space
        #action_size: dimension of the action space
        #device: torch.device()-> cuda or cpu
        #random_seed: used for reproducability
        #add_noise: boolean parameter used to indicate if OUNoise is added

        #agent parameters
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.add_noise = add_noise
        self.seed = random.seed(random_seed)
        #actor and critic models
        self.actor_local = ActorDDPG(state_size, action_size).to(device)
        self.actor_target = ActorDDPG(state_size, action_size).to(device)
        #actor optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = CriticDDPG(state_size, action_size).to(device)
        self.critic_target = CriticDDPG(state_size, action_size).to(device)
        #critic optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        #noise process
        self.noise = OUNoise(action_size, random_seed, sigma = 0.05)
        #replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device, random_seed=random_seed)

    def step(self, state, action, reward, next_state, done):
        # Save experience / reward

        self.memory.add(state, action, reward, next_state, done)
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)


    def act(self, state):
        #return action (clipped) for given state
        state = torch.from_numpy(state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        #reset noise model
        self.noise.reset()


    def learn(self, experiences, gamma):
        #experiences: Tuple[torch.tensor]
        states, actions, rewards, next_states, dones = experiences

        # update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #soft update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def train_agent(self, env,  n_episodes=1000, max_t=300, print_every=100):

        scores_deque = deque(maxlen=print_every)
        scores = []
        for i_episode in range(1, n_episodes+1):
            state = env.reset()[0]
            self.reset()
            score = 0
            for t in range(max_t):
                action = self.act(state)

                step_action = env.step(action)
                next_state, reward, done, _ , _= step_action
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_deque.append(score)
            scores.append(score)
            self.n = i_episode
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
            if i_episode % print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= 200.0:
                print('\nEnvironment solved in {:d} episodes\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                break
        #save actor and critic models at the end of training
        torch.save(self.actor_local.state_dict(), 'checkpoint_ddpg_actor.pth')
        torch.save(self.critic_local.state_dict(), 'checkpoint_ddpg_critic.pth')
        return scores


    def soft_update(self, local_model, target_model, tau):
        #tau: interpolation parameter
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def load(self,file_actor, file_critic):
        #load saved model parameters
        self.actor_local.load_state_dict(torch.load(file_actor, map_location=self.device))
        self.critic_local.load_state_dict(torch.load(file_critic, map_location=self.device))
