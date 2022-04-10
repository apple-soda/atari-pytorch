import numpy as np
import torch

class ExperienceReplay:
    def __init__(self, size):
        self.memory = []
        self.size = size
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        # encode dones to 1
        done = int(not done)
        data = (state, action, reward, next_state, done)
        if (self.position >= len(self.memory)):
            self.memory.append(data)
        else:
            self.memory[self.position] = data
        self.position = (self.position + 1) % self.size
        
    def get(self, batch_size):
        indices = np.random.randint(0, len(self.memory) - 1, batch_size)
        batch = [self.memory[i] for i in indices]
        return batch
    
    def __len__(self):
        return len(self.memory)