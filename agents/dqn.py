import random
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.replay import *
from networks.cnn import * 

class DQN:
    def __init__(self, observation_space, action_space, **params):
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # greeks
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        
        # agent
        self.memory_size = params['memory_size']
        
        # network 
        self.device = torch.device(params['device'])
        self.network = params['network']
        self.batch_size = params['batch_size']
        self.tnet_update = params['tnet_update']
        
        # default
        self.dtype = torch.float32
        
        self._build_agent()
        
    def _build_agent(self): # todo : let user specify his/her own loss
        self.memory = ExperienceReplay(self.memory_size)
        self.network = self.network.to(self.device)
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.alpha)
        self.loss = nn.SmoothL1Loss().to(self.device)
        
        # double dqn
        if self.tnet_update is not None:
            self.target_network = copy.deepcopy(self.network)
        
        self.optim_steps = 0
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
    def action(self, state):
        if (random.random() <= self.epsilon):
            action = self.action_space.sample()
        else:
            state = torch.tensor(state, device=self.device, dtype=self.dtype)
            action_values = self.network(state.unsqueeze(0)).detach().cpu().numpy()
            action = np.argmax(action_values)
            
        return action
    
    def update(self):
        minibatch = self.memory.get(self.batch_size)
        
        # unpack tuples and transform to tensors
        # actions and dones should be integers
        states = np.stack([i[0] for i in minibatch])
        next_states = np.stack(i[3] for i in minibatch)
        states = torch.tensor(states, dtype=self.dtype, device=self.device)
        next_states = torch.tensor(next_states, dtype=self.dtype, device=self.device)

        rewards = torch.tensor([i[2] for i in minibatch], dtype=self.dtype, device=self.device)
        actions = torch.tensor([i[1] for i in minibatch], device=self.device).unsqueeze(dim=1)
        dones = torch.tensor([i[4] for i in minibatch], device=self.device)
        
        self.optim.zero_grad()
        
        # update target network
        if (self.optim_steps % self.tnet_update == 0):
            self.target_network.load_state_dict(self.network.state_dict())
        
        Q_predicted = torch.gather(self.network(states), 1, actions).squeeze()
        Q_s1 = self.network(next_states) # a1 is always chosen by original Q network
        a1 = Q_s1.argmax(dim=1).reshape(-1, 1)
        Q_target = self.target_network(next_states)
        Q_actual = rewards + self.gamma * torch.gather(Q_target, 1, a1).squeeze() * dones
        
        loss = self.loss(Q_predicted, Q_actual) # input, output
        loss.backward()
        self.optim.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.optim_steps += 1
        
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