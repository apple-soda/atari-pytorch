import random
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from core.replay import *
from networks.cnn import * 

class DQNAgent:
    def __init__(self, observation_space, action_space, **params):
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # agent parameters
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.memory_size = params['memory_size']
        self.num_model_updates = 0
        self.target_net_updates = params['target_net_updates']
        
        # network
        self.device = torch.device(params['device'])
        self.batch_size = params['batch_size']
        self.dtype = torch.float32
        
        # _build_agent()
        self.replay = ExperienceReplay(self.memory_size)
        self.network = params['network'].to(self.device)
        self.target_network = params['network'].to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optim = optim.Adam(self.network.parameters(), lr=self.alpha)
        self.loss = torch.nn.SmoothL1Loss().to(self.device)
        
    def action(self, state):
        if (random.random() <= self.epsilon):
            action = self.action_space.sample()
        else:
            state = torch.tensor(state, device=self.device, dtype=self.dtype)
            action_values = self.network(state.unsqueeze(0) / 255).detach().cpu().numpy()
            action = np.argmax(action_values)
            
        return action
        
    def remember(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)
                                      
    def update(self, update=1):
        for e in range(update):
            self.optim.zero_grad()

            minibatch = self.replay.get(self.batch_size)

            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).to(self.device).to(self.dtype)) / 255

            actions = np.stack([i[1] for i in minibatch])
            actions = torch.tensor(actions).to(self.device).to(torch.int64)
            rewards = torch.tensor([i[2] for i in minibatch]).to(self.device)

            # uint8 to float32 and normalize to 0-1
            next_obs = (torch.stack([i[3] for i in minibatch]).to(self.device).to(self.dtype)) / 255

            dones = torch.tensor([i[4] for i in minibatch]).to(self.device)

            Q_predicted = torch.gather(self.network(obs), 1, actions.unsqueeze(dim=1)).squeeze()
            Q_s1 = self.network(next_obs)
            a1 = Q_s1.argmax(dim=1).reshape(-1, 1)
            Q_prime_s1 = self.target_network(next_obs)
            Q_actual = rewards + self.gamma * torch.gather(Q_prime_s1, 1, a1).squeeze() * dones

            # loss
            loss = self.loss(Q_predicted, Q_actual) # torch.mean(torch.pow(obs_Q - target, 2))
            loss.backward()
            self.optim.step()

            self.num_model_updates += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if self.num_model_updates%self.target_net_updates == 0:
                self.target_network.load_state_dict(self.network.state_dict())
                
    def save(self, path):
        torch.save({'network': self.network.state_dict(),
                    'optim': self.optim.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        
        self.network.load_state_dict(checkpoint['network'])
        self.optim.load_state_dict(checkpoint['optim'])
        
        if self.target_update_freq is not None:
            self.target_network = copy.deepcopy(self.network)
        
        self.epsilon = .01 # for testing