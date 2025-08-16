# @title Neccessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from models import Actor, Critic
# @title A2C

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.0005       # learning rate of the actor
LR_CRITIC = 0.0005      # learning rate of the critic

class A2CAgent:
    def __init__(self, state_size, action_size, device):
        #INPUT:
        # state_size: dimension of the observation space
        # action_size: dimension of the action space
        # device: torch.device()->cuda or cpu
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        #networks
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size).to(self.device)
        self.target_actor = Actor(self.state_size, self.action_size).to(self.device)
        self.target_critic = Critic(self.state_size).to(self.device)
        #optimizers for actor and critic models
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        #lists used for training
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

        self._soft_update(self.target_actor, self.actor, 1.0)
        self._soft_update(self.target_critic, self.critic, 1.0)

    def _soft_update(self, target, source, tau):
        #tau: interpolation parameter
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def calculate_returns(self, rewards, normalize = False):
        #calculating returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R * GAMMA
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        #normalize returns
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def calculate_advantages(self, returns, values, normalize = True):
        #calculate advantages
        advantages = returns - values
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def train(self, env, max_t=500):

        log_prob_actions = []
        values = []
        rewards = []
        done = False
        episode_reward = 0

        state = env.reset()[0]

        for t in range(max_t):

            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            #get actor logits
            action_pred = self.actor(state)
            #get critic Q-values
            value_pred = self.critic(state)
            #convert actor logits to probabilities
            action_prob = F.softmax(action_pred, dim=-1)
            #create a discrete probaility distibution
            dist = torch.distributions.Categorical(action_prob)
            #sample
            action = dist.sample()
            #find log probabilities
            log_prob_action = dist.log_prob(action)

            next_state, reward, done,_ , _ = env.step(action.item())

            log_prob_actions.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)

            episode_reward += reward
            state = next_state

            if done:
                break

        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values).squeeze(-1)
        #calculate returns
        returns = self.calculate_returns(rewards).to(self.device)
        #calculate advantages
        advantages = self.calculate_advantages(returns, values)
        #calculate policy and value losses
        policy_loss, value_loss = self.update_policy(advantages, log_prob_actions, returns, values)
        #soft update the networks
        self._soft_update(self.target_actor, self.actor, TAU)
        self._soft_update(self.target_critic, self.critic, TAU)

        return policy_loss, value_loss, episode_reward

    def update_policy(self, advantages, log_prob_actions, returns, values):

        advantages = advantages.detach()
        returns = returns.detach()
        #policy loss
        policy_loss = - (advantages * log_prob_actions).sum()
        #value loss
        value_loss = F.smooth_l1_loss(values.float(), returns.float()).sum()
        #additional entropy term for exploration(stabilizes training)
        entropy = 0.01 * (-log_prob_actions.exp() * log_prob_actions).sum()
        policy_loss -= entropy
        #optimize
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        policy_loss.backward()
        value_loss.backward()
        #clip the gradient norm to unit size
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        return policy_loss.item(), value_loss.item()

    def train_agent(self, env, num_episodes=1000, max_t=500):
        scores = []
        train_rewards = deque(maxlen=100)
        for episode in range(1, num_episodes + 1):
            policy_loss, value_loss, train_reward = self.train(env, max_t=max_t)

            train_rewards.append(train_reward)
            scores.append(train_reward)
            mean_train_rewards = np.mean(train_rewards)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, mean_train_rewards), end="")
            if episode % 100 == 0:
                print(f' | Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:7.1f}')
            if mean_train_rewards>=200:
                print("Environment Solved")
                break
        torch.save(self.actor.state_dict(), 'checkpoint_actor.pth')
        torch.save(self.critic.state_dict(), 'checkpoint_critic.pth')
        return scores

    def load(self, file_actor, file_critic) :
        self.actor.load_state_dict(torch.load(file_actor, map_location = self.device))
        self.critic.load_state_dict(torch.load(file_critic, map_location=self.device))
