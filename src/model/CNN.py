from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall
from model.base import BaseModel

# class CNN_model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1,6,5) #解释：输入通道数为1，输出通道数为6，卷积核大小为5
#         self.pool = nn.MaxPool2d(2,2) #解释：池化核大小为2，步长为2
#         self.conv2 = nn.Conv2d(6,16,5) #解释：输入通道数为6，输出通道数为16，卷积核大小为5
#         self.fc1 = nn.Linear(16 * 5 * 5, 120) #解释：输入通道数为16*5*5，输出通道数为120
#         self.fc2 = nn.Linear(120, 84) #解释：输入通道数为120，输出通道数为84
#         self.fc3 = nn.Linear(84, 2) #解释：输入通道数为84，输出通道数为2

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) #解释：将多维的输入一维化，即将[16,5,5]变成[1,400]
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class CNN(BaseModel):
    def __init__(self, learning_rate = 0.05):
        super().__init__(learning_rate=learning_rate)
        self.save_hyperparameters()  # This will save all passed hyperparameters to self.hparams
        
        self.conv1 = nn.Conv2d(1,6,5) #解释：输入通道数为1，输出通道数为6，卷积核大小为5
        self.pool = nn.MaxPool2d(2,2) #解释：池化核大小为2，步长为2
        self.conv2 = nn.Conv2d(6,16,5) #解释：输入通道数为6，输出通道数为16，卷积核大小为5
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #解释：输入通道数为16*5*5，输出通道数为120
        self.fc2 = nn.Linear(120, 84) #解释：输入通道数为120，输出通道数为84
        self.fc3 = nn.Linear(84, 2) #解释：输入通道数为84，输出通道数为2
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) #解释：将多维的输入一维化，即将[16,5,5]变成[1,400]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
