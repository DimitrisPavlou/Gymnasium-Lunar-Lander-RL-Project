# @title Neccessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
# @title Models

# Q-network used in DQNAgent and DoubleDQNAgent

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, random_seed, fc1_units = 512, fc2_units = 256, batch_norm = False):

        #simple Q-network with or without batch normalizaition
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(random_seed)
        if batch_norm:
            self.qnet = nn.Sequential(
                nn.Linear(state_size, fc1_units),
                nn.BatchNorm1d(fc1_units),
                nn.ReLU(),
                nn.Linear(fc1_units, fc2_units),
                nn.BatchNorm1d(fc2_units),
                nn.ReLU(),
                nn.Linear(fc2_units, action_size)
            )

        else:
            self.qnet = nn.Sequential(
                nn.Linear(state_size, fc1_units),
                nn.ReLU(),
                nn.Linear(fc1_units, fc2_units),
                nn.ReLU(),
                nn.Linear(fc2_units, action_size)
            )

    def forward(self, state):
        return self.qnet(state)

#======================================================================================

# Dueling Q-network used in DuelingDQNAgent and DuelingDoubleDQNAgent

class DuelingQNetwork(nn.Module):

    def __init__(self, state_size, action_size, fc1_size = 256, fc2_size = 256):

        #simple Dueling Q-network
        super(DuelingQNetwork, self).__init__()
        #feature stream
        self.features = nn.Sequential(
            nn.Linear(state_size , fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU()
        )
        #value stream
        self.value_network = nn.Sequential(
            nn.Linear(fc2_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, 1)
        )
        #advantage stream
        self.advantage_network = nn.Sequential(
            nn.Linear(fc2_size , fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size , action_size)
        )

    def forward(self, state):
        # map state -> action values."""
        x = self.features(state)
        vals = self.value_network(x)
        adv = self.advantage_network(x)

        qvalues = vals + (adv - adv.mean())
        return qvalues
#======================================================================================


# Actor and Critic Networks used in DDPGAgent

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorDDPG(nn.Module):

    def __init__(self, state_size, action_size, fc1_units=400, fc2_units=200):
        #actor model for ddpg
        super(ActorDDPG, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        #actor (policy) network that maps states -> actions.
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class CriticDDPG(nn.Module):

    def __init__(self, state_size, action_size, fcs1_units=400, fc2_units=200):
        #critic model of DDPG
        super(CriticDDPG, self).__init__()
        self.fcs1 = nn.Linear(state_size + action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


    def forward(self, state, action):
        #critic (value) network that maps (state, action) pairs -> Q-values
        x = torch.cat((state, action), dim = 1)
        x = F.relu(self.fcs1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


#======================================================================================

# basic actor and critic models

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        #basic actor model
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_size):
        #basic critic model
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)


# combined Actor Critic model for PPO Agent

class ActorCriticPPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticPPO, self).__init__()

        self.actor = Actor(state_dim , action_dim)
        self.critic = Critic(state_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # device used

    def act(self, state):

        state = torch.FloatTensor(state.cpu()).to(self.device)

        action_logits = self.actor(state)# output of actor nn

        SoftMax = nn.Softmax(dim=-1)
        action_probs = SoftMax(action_logits)# outputs -> probabilities

        dist = Categorical(probs = action_probs)# distribution according to the probabilities
        selected_action = dist.sample()# random action sampling from the distribution

        log_prob = dist.log_prob(selected_action)# log of the probability of the seleced action (for calculating loss)

        state_value = self.critic(state)# expected value of the current

        return selected_action , log_prob, state_value


    def evaluate(self, state, action):

        state = torch.FloatTensor(state.cpu()).to(self.device)

        action_logits = self.actor(state)# output of actor

        SoftMax = nn.Softmax(dim=-1)
        action_probs = SoftMax(action_logits)# outputs -> probabilities

        dist = Categorical(probs = action_probs)# discrete distribution from the probabilities

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs , state_values, dist_entropy

    def save_model(self, filepath: str):
        torch.save({
            'Actor_state_dict': self.actor.state_dict(),
            'Critic_state_dict': self.critic.state_dict(),
        }, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['Actor_state_dict'])
        self.critic.load_state_dict(checkpoint['Critic_state_dict'])