# @title Neccessary imports
import torch
import numpy as np
from collections import namedtuple, deque
import random
import copy

# @title Helper Classes
class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size,device, random_seed):
        #INPUT
        # action_size: dimension of the action space
        # buffer_size: max length of the buffer
        # batch_size: length of the mini batches of experiences
        # device: torch.device() -> cuda or cpu
        # random_seed: used for reproducability

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        self.seed = random.seed(random_seed)

    def add(self, state, action, reward, next_state, done):
        # add a new experience to memory
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        # randomly sample a batch of experiences from memory
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        # return the current size of internal memory
        return len(self.memory)


class PrioritizedExperienceReplayBuffer:

    def __init__(self, buffer_size, batch_size, alpha, device, random_seed, beta_start=0.4, beta_frames=100000):
        #INPUT
        # buffer_size: max length of buffer
        # batch_size: length of the mini batches of experiences
        # alpha: parameer for the calculation of probabilities
        # beta_start: starting value of beta for correction weights
        # beta_frames: parameter used for incrementing beta

        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames  # total frames to reach max beta
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.pos = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.buffer = []
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        self.seed = random.seed(random_seed)

    def add(self, state, action, reward, next_state, done):
        # add experience to buffer with priority
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(self.experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = self.experience(state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, beta=None):
        #sample experience from the buffer
        if beta is None:
            beta = self.beta
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        # create the probability distribution
        probs = prios ** self.alpha
        probs /= probs.sum()
        #finde indices
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        #calculate the correction weights
        weights = (total * probs[indices]) ** (-beta)
        #normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return indices, (states, actions, rewards, next_states, dones), weights

    def update_priorities(self, batch_indices, batch_priorities):
        #update priorities from training
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def increment_beta(self, frame_idx):
        #increment beta during training
        self.beta = min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self):
        return len(self.buffer)



class OUNoise:
    #Ornstein-Uhlenbeck process

    def __init__(self, size, seed, mu=0., theta=0.1, sigma=0.2):
        #Initialize parameters and noise process
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        #Reset the internal state (= noise) to mean (mu)
        self.state = copy.copy(self.mu)

    def sample(self):
        #Update internal state and return it as a noise sample
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state