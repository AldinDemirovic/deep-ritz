import torch
import torch.nn as nn
import torch.nn.functional as F
width = 32

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(2, width)
        self.lin2 = nn.Linear(width, width)
        self.lin3 = nn.Linear(width, width)
        self.lin4 = nn.Linear(width, width)
        self.lin5 = nn.Linear(width, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = self.lin5(x)
        return x

