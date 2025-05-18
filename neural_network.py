import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermarkDetectorCNN(nn.Module):
    def __init__(self):
        super(WatermarkDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Для входа 256x256
        self.fc2 = nn.Linear(128, 4)  # Выход: (x1, y1, x2, y2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x