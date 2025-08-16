# @title Neccessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from models import QNetwork
from Buffers import PrioritizedExperienceReplayBuffer, ReplayBuffer



# @title DQN
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
PRIORITY_SCALE = 0.7

class DQNAgent:

    def __init__(self, state_size, action_size, device, random_seed,hidden_sizes = [512, 256], batch_norm = False, per = False):
        # INPUT:
        # state_size: Dimension of the observation space
        # action_size: Dimension of the action space
        # device: torch.device() object -> cuda or cpu
        # hidden_sizes: number of neurons per layer of the Q-network (by default 2 hidden layers are used)
        # batch_norm: Boolean variable indicating if batch normalization is applied to the Q networks
        # per: Boolean variable indicating if Prioritized Experience Replay is used

        #initialize agent parameters
        self.state_size = state_size
        self.action_size = action_size
        self.per = per
        self.epsilon = 1.0
        self.device = device
        self.seed = random.seed(random_seed)
        self.beta_frames = 100000  # Total frames to reach max beta for Prioritized Experience
        self.n = 0  # To keep track of frames
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, random_seed,fc1_units=hidden_sizes[0], fc2_units = hidden_sizes[1],batch_norm=batch_norm).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size,random_seed, fc1_units=hidden_sizes[0], fc2_units = hidden_sizes[1],batch_norm=batch_norm).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        #optimizer
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory
        if per:
            self.memory = PrioritizedExperienceReplayBuffer(batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, alpha = PRIORITY_SCALE, random_seed=random_seed, device=device)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device, random_seed=random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            self.n += 1
            # check if Prioritized experience replay is used
            if self.per:
                if len(self.memory) > BATCH_SIZE:
                    self.memory.increment_beta(self.n)
                    indices , experiences , normalized_weights = self.memory.sample(beta=self.memory.beta)
                    self.learn_PER(indices, experiences, normalized_weights, GAMMA)
            # If enough samples are available in memory, get random subset and learn
            else:
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        # return action for given state

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn_PER(self, indices, experiences, sampling_weights , gamma):
        # update parameters using Prioritized Experience Replay

        states, actions, rewards, next_states, dones = experiences
        #calculate current Q from local network
        Q_current = self.qnetwork_local(states).gather(1, actions.type(torch.int64))
        #calculate target Q of the next state from target network
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #calculate expected Q values from the Bellman equation
        Q_expected = rewards + gamma * Q_target_next * (1 - dones)

        # calculate the difference
        error = Q_expected - Q_current

        priorities = error.abs().cpu().detach().numpy().flatten() # priority = |error|
        self.memory.update_priorities(indices, priorities + 1e-6)  # priority > 0 so we add an offset
        # correction weights to avoid bias
        Sampling_Weights = (torch.Tensor(sampling_weights).view((-1, 1))).to(self.device)

        # compute loss
        loss = F.smooth_l1_loss(Q_expected*Sampling_Weights, Q_current*Sampling_Weights)
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def learn(self, experiences, gamma):
        #update parameters

        #experiences : Tuple[torch.tensor]
        states, actions, rewards, next_states, dones = experiences
        #calculate current Q from local network
        Q_current = self.qnetwork_local(states).gather(1, actions.type(torch.int64))
        #calculate target Q of the next state from target network
        Q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #calculate expected Q values from the Bellman equation
        Q_expected = rewards + gamma * Q_target_next * (1 - dones)

        # Compute loss
        loss = F.smooth_l1_loss(Q_expected, Q_current)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        # tau : interpolation parameter
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train_agent(self, env, n_episodes = 2000 , max_t = 500, eps_start=1.0, eps_end=0.01, eps_decay=0.995) :
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon

        for i_episode in range(1, n_episodes+1):
            state = env.reset()[0]
            score = 0
            for t in range(max_t):
                action = self.act(state, eps)
                step_action = env.step(action)
                next_state, reward, terminated, truncated , _= step_action
                done = terminated or truncated
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward

                if done:
                    break

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            self.epsilon = eps
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                break
        #save model at the end of training
        torch.save(self.qnetwork_local.state_dict(), "checkpoint_dqn_bn_per.pth")
        return scores

    def load(self, file):
        #load saved model parameters
        self.qnetwork_local.load_state_dict(torch.load(file, map_location=self.device))