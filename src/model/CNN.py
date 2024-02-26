from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall
from model.base import BaseModel


class CNN(BaseModel):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adaptive pooling allows us to specify the output size we want for the feature maps
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(-1, 256 * 6 * 6)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#       test_accuracy         0.8876736164093018
#          test_f1            0.8790778517723083
#         test_loss           0.2728959321975708
#      test_precision         0.8321838974952698
#        test_recall           0.932607114315033
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


# class CNN(BaseModel):

#     def __init__(self, learning_rate):
#         super().__init__()
#         self.learning_rate = learning_rate
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         # Adaptive pooling allows us to specify the output size we want for the feature maps
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

#         # Fully connected layers
#         self.fc1 = nn.Linear(512 * 6 * 6, 2048)
#         self.fc5 = nn.Linear(2048, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512,128)
#         self.fc4 = nn.Linear(128, 2)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = self.pool(F.relu(self.bn5(self.conv5(x))))
#         x = self.adaptive_pool(x)
#         x = x.view(-1, 512 * 6 * 6)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#       test_accuracy         0.8988281488418579
#          test_f1            0.8867621421813965
#         test_loss           0.2610653042793274
#      test_precision         0.8706926107406616
#        test_recall          0.9040333032608032
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────