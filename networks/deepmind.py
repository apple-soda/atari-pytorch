import torch
import torch.nn as nn
import torch.nn.functional as F

# predefined networks
class DeepmindCNN(nn.Module):
    def __init__(self):
        super().__init__()

        network = [
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        ]
        
        self.network = nn.Sequential(*network)
        
    def forward(self, x): 
        actions = self.network(x)
        return actions