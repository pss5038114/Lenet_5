import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: 입력 1채널 -> 출력 6채널, 5x5 커널
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # C3: 입력 6채널 -> 출력 16채널, 5x5 커널
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # C5
        self.fc2 = nn.Linear(120, 84)         # F6
        self.fc3 = nn.Linear(84, 10)          # Output (0~9)

    def forward(self, x):
        # S2: Max Pooling (원문은 Avg지만 현대적 성능을 위해 Max 사용 가능, 여기선 전통에 가깝게 구현)
        x = F.tanh(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # S4: Avg Pooling
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten: 2D 특징 맵을 1D 벡터로 변환
        x = x.view(-1, 16 * 5 * 5)
        
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x