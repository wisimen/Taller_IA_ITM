import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.functional as F

class FruitsNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=20, kernel_size=5)
        self.fc1 = nn.Linear(9680, 9680)
        self.fc2 = nn.Linear(9680, 6550)
        self.fc3 = nn.Linear(6550, 131)
        # self.fc1 = nn.Linear(24200, 24200)
        # self.fc2 = nn.Linear(24200, 17424)
        # self.fc3 = nn.Linear(17424, 131)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
