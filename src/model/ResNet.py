from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model.base import BaseModel
import torchvision.models as models



class ResNet18Model(BaseModel):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        # 加载预训练的ResNet18模型
        self.resnet18 = models.resnet18(pretrained=True)
        # 替换最后的全连接层以匹配目标类别数
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        return self.resnet18(x)


