import torch
import torch.nn as nn
import torch.nn.functional as F

# flexible network framework 
class Net(nn.Module):
    def __init__(self, *network):
        super().__init__()
        self.network = nn.Sequential(*network)
        
    def forward(self, x): 
        actions = self.network(x)
        return actions
    
