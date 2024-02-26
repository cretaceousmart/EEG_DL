import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall


class BaseModel(pl.LightningModule):
  """
  This is the base model class for most of the model in this project (CNN, DenseNet, etc.).
  It define the functions include:

  
  forward: Forward pass of the model. when you call self(x) it will automatically call this function.

  traning_step: Perform a training step on the provided batch.
  test_step: Perform a test step on the provided batch.
  validation_step: Perform a validation step on the provided batch.
   
  configure_optimizers: Lightning optimizers configuration. Uses Adam with 0.1 initial

  """

  def __init__(self, learning_rate: float = 0.1, patience: int = 10):
    """
    Initialize the model.

    Args:
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.1.
    """
    super().__init__()
    self.learning_rate = learning_rate
    self.patience = patience

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

    # 为什么要写loss_fn而不是loss：因为loss是pytorch_lightning的一个属性，如果写成loss，会报错
    self.loss_fn = nn.CrossEntropyLoss()


  def forward(self,x) -> torch.Tensor:
    raise NotImplementedError("This is an abstract method and should be implemented by subclass.")


  def training_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      pred = torch.argmax(logits, dim=1)

      loss = self.loss_fn(logits, y)
      self.log('train_loss', loss)

      accuracy = self.train_accuracy(pred, y)
      self.log('train_accuracy', accuracy)

      f1 = self.train_f1(pred, y)
      self.log('train_f1', f1)

      precision = self.train_precision(pred, y)
      self.log('train_precision', precision)

      recall = self.train_recall(pred, y)
      self.log('train_recall', recall)

      return loss

  
  def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      pred = torch.argmax(logits, dim=1)

      loss = self.loss_fn(logits, y)
      self.log('val_loss', loss)

      accuracy = self.val_accuracy(pred, y)
      self.log('val_accuracy', accuracy)

      f1 = self.val_f1(pred, y)
      self.log('val_f1', f1)

      precision = self.val_precision(pred, y)
      self.log('val_precision', precision)
      
      recall = self.val_recall(pred, y)
      self.log('val_recall', recall)

      return loss

  def test_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      pred = torch.argmax(logits, dim=1)

      loss = self.loss_fn(logits, y)
      self.log('test_loss', loss)

      accuracy = self.test_accuracy(pred, y)
      self.log('test_accuracy', accuracy)

      f1 = self.test_f1(pred, y)
      self.log('test_f1', f1)

      precision = self.test_precision(pred, y)
      self.log('test_precision', precision)

      recall = self.test_recall(pred, y)
      self.log('test_recall', recall)

      return loss


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=self.patience),  # 如果需要10次没有改进才降低学习率，则这里应为patience=10
        "monitor": "train_loss",  
        "interval": "epoch",
        "frequency": 1  # 每个 epoch 检查一次
    }
    return [optimizer], [scheduler]  # scheduler 需要被放在列表中

