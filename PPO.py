# @title Neccessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import ActorCriticPPO

# @title PPO Discrete
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 0.0005       # learning rate of the actor
LR_CRITIC = 0.0005      # learning rate of the critic



class PPOAgent:

  def __init__(self , state_size, action_size, device):
        #INPUT:
        # state_size: dimension of the observation space
        # action_size: dimension of the action space
        # device: torch.device() -> cuda or cpu
        self.max_training_timesteps = int(5e5)     # break training loop if timeteps > max_training_timesteps
        self.update_timestep = 1000                # update policy every n timesteps
        self.epochs = 10                           # update policy for #epochs
        self.epsilon = 0.2                         # clip parameter for PPO
        self.gamma = GAMMA                         # discount factor
        self.device = device

        self.state_dim = state_size
        self.action_dim = action_size
        #lists used in training
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

        #create a common model of Actor-Critic
        self.policy = ActorCriticPPO(self.state_dim, self.action_dim).to(self.device)
        #create a common optimizer for the common model
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': LR_ACTOR},
                        {'params': self.policy.critic.parameters(), 'lr': LR_CRITIC}
                    ])

        self.policy_old = ActorCriticPPO(self.state_dim, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

  def empty_lists(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]

  def act(self, state):
        #return action given a state
        with torch.no_grad():
          state = torch.FloatTensor(state).to(self.device)
          action, action_logprob, state_val = self.policy_old.act(state)

        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.state_values.append(state_val)
        return action.item()

  def update(self , rewards):
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # calculating ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # calculating Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * F.smooth_l1_loss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

  def learn(self):

        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        self.update(rewards)
        self.empty_lists()

  def train_agent(self, env, num_episodes = 2000, max_t = 600):
        time_step = 0
        i_episode = 1
        scores = []

        for i_episode in range(1, num_episodes+1):
            state = env.reset()[0]
            current_ep_reward = 0
            for t in range(1, max_t+1):
                action = self.act(state)
                state, reward, terminated , truncated, _ = env.step(action)
                done = terminated or truncated
                self.rewards.append(reward)
                self.dones.append(done)
                time_step +=1
                current_ep_reward += reward
                if time_step % self.update_timestep == 0:
                    self.learn()
                if done:
                    break
            scores.append(current_ep_reward)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores[-100:])), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores[-100:])))
            if np.mean(scores[-100:]) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,np.mean(scores[-100:])))
                break
        return scores

  def save(self, filepath: str):
    self.policy.save_model(filepath)

  def load(self, filepath: str):
    self.policy.load_model(filepath)
