import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # 动态计算 fc1 的输入大小
        dummy_input = torch.zeros(1, 3, 224, 224)  # 假设输入图片大小为 224x224
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        self.fc1_input_size = x.view(x.size(0), -1).size(1)

        self.fc1 = nn.Sequential(
            nn.Linear(self.fc1_input_size, 1000),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.fc2(x)

        return x
