import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    DQN Agent, uses a convolutional neural network to approximate Q(s,a).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 2)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, 2)
        self.batch3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(32*59*59, 3)
        
    def forward(self, x):
        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
        x = F.relu(self.batch3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))