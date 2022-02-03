import torch
import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=(16 * 5 * 5), out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))  # Convolution + ReLU + Max Pooling
        x = self.pool(f.relu(self.conv2(x)))  # Convolution + ReLU + Max Pooling
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = f.relu(self.fc1(x))  # Feed forward + ReLU
        x = f.relu(self.fc2(x))  # Feed forward + ReLU
        x = self.fc3(x)  # Feed forward
        return x
