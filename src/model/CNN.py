from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
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


class CNN(pl.LightningModule):
    def __init__(self, learning_rate = 0.05):
        super().__init__()
        self.save_hyperparameters() #解释：保存超参数，方便在训练过程中调用
        self.model = CNN_model()
        self.learning_rate = learning_rate

        # 初始化度量
        self.train_accuracy = Accuracy(task='binary', num_classes=2)
        self.val_accuracy = Accuracy(task='binary', num_classes=2)
        self.test_accuracy = Accuracy(task='binary', num_classes=2)
        self.train_f1 = F1Score(task='binary', num_classes=2)
        self.val_f1 = F1Score(task='binary', num_classes=2)
        self.test_f1 = F1Score(task='binary', num_classes=2)
        self.train_precision = Precision(task='binary', num_classes=2)
        self.val_precision = Precision(task='binary', num_classes=2)
        self.test_precision = Precision(task='binary', num_classes=2)
        self.train_recall = Recall(task='binary', num_classes=2)
        self.val_recall = Recall(task='binary', num_classes=2)
        self.test_recall = Recall(task='binary', num_classes=2)



        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        accuracy = self.train_accuracy(logits, y)
        self.log('train_accuracy', accuracy)
        f1 = self.train_f1(logits, y)
        self.log('train_f1', f1)
        precision = self.train_precision(logits, y)
        self.log('train_precision', precision)
        recall = self.train_recall(logits, y)
        self.log('train_recall', recall)

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss)
        accuracy = self.val_accuracy(logits, y)
        self.log('val_accuracy', accuracy)
        f1 = self.val_f1(logits, y)
        self.log('val_f1', f1)
        precision = self.val_precision(logits, y)
        self.log('val_precision', precision)
        recall = self.val_recall(logits, y)
        self.log('val_recall', recall)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('test_loss', loss)
        accuracy = self.test_accuracy(logits, y)
        self.log('test_accuracy', accuracy)
        f1 = self.test_f1(logits, y)
        self.log('test_f1', f1)
        precision = self.test_precision(logits, y)
        self.log('test_precision', precision)
        recall = self.test_recall(logits, y)
        self.log('test_recall', recall)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer