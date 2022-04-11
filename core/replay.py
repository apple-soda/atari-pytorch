import numpy as np
import torch

class ExperienceReplay:
    def __init__(self, size):
        self.memory = []
        self.size = size
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        # transforms into 1s and 0s : 0 -> terminal state
        done = int(not done) 
        state, next_state = torch.tensor(state), torch.tensor(next_state)
        data = (state, action, reward, next_state, done)
        if (self.position >= len(self.memory)):
            self.memory.append(data)
        else:
            self.memory[self.position] = data
        self.position = (self.position + 1) % self.size
    
    def get(self, batch_size):
        if len(self.memory) - 1 == 0:
            indices = np.zeros(batch_size).astype(int)
        else:
            indices = np.random.randint(0, len(self.memory) - 1, batch_size)
        batch = [self.memory[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.memory)
    
class GPUExperienceReplay:
    def __init__(self, size, device):
        self.memory = []
        self.size = size
        self.position = 0
        self.device = torch.device(device)
        self.dtype = torch.float32
        
    def add(self, state, action, reward, next_state, done):
        done = int(not done) 
        state = torch.tensor(state, device=self.device, dtype=self.dtype)
        action = torch.tensor(action, device=self.device, dtype=torch.int64)
        reward = torch.tensor(reward, device=self.device)
        next_state = torch.tensor(next_state, device=self.device, dtype=self.dtype)
        done = torch.tensor(done, device=self.device)
        
        data = (state, action, reward, next_state, done)
        
        if (self.position >= len(self.memory)):
            self.memory.append(data)
        else:
            self.memory[self.position] = data
        self.position = (self.position + 1) % self.size
        
    def get(self, batch_size):
        if len(self.memory) - 1 == 0:
            indices = np.zeros(batch_size).astype(int)
        else:
            indices = np.random.randint(0, len(self.memory) - 1, batch_size)
        batch = [self.memory[i] for i in indices]
        return batch
        
    def __len__(self):
        return len(self.memory)
    
    