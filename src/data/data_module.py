from typing import List, Tuple, Union
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import pytorch_lightning as pl
import pandas as pd
import os
from PIL import Image



class EEG_Dataset(Dataset):
    def __init__(self, eeg_file_names, test_mode=False, transform=None):
        self.all_images, self.all_labels = self.prepare_data_from_multi_file(eeg_file_names, test_mode)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.all_images)

    def __getitem__(self, index: int):
        # 使用 PIL 加载图像，而不是直接使用 numpy 数组
        image_path = self.all_images[index]
        image = Image.open(image_path).convert('L')  # 转换为灰度图像

        # 应用传递给类的转换
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.all_labels[index], dtype=torch.int64)
        return image, label

    def prepare_data_from_multi_file(self, eeg_file_names, test_mode):
        all_images = []
        all_labels = []

        for eeg_file_name in eeg_file_names:
            # 读取每个图像的标签
            with pd.ExcelFile(rf'../src/data/test_data/{eeg_file_name}/{eeg_file_name}.xlsx') as xls:
                label_df = pd.read_excel(xls, 'Sheet1', header=None)
                label_df.rename(columns={0: 'label'}, inplace=True)

            if test_mode:
                label_df = label_df[:100]  # 如果是测试模式，仅使用部分数据

            # 读取图像数据
            image_folder = rf'../src/data/output_image/{eeg_file_name}/'
            label_size = 3600 if not test_mode else 100 

            for image_id in range(label_size):
                image_path = os.path.join(image_folder, f'{eeg_file_name}_{image_id}.png')
                all_images.append(image_path)

            # 将标签追加到总列表
            all_labels.extend(label_df['label'].tolist())

        # 将标签转换为 numpy 数组
        all_labels_np = np.array(all_labels)

        return all_images, all_labels_np



class EEG_DataModule(pl.LightningDataModule):

    def __init__(self,
                 eeg_file_names: List[str],
                 test_mode: bool = False,
                 image_size: int = 128,
                 batch_size: int = 32,
                 train_size: float = 0.7, val_size: float = 0.1, test_size: float = 0.2,
                 ):
        super().__init__()
        self.eeg_file_names = eeg_file_names
        self.test_mode = test_mode
        
        self.image_size = image_size
        self.batch_size = batch_size

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        
        #图片的归一化处理，原因是：在训练过程中，如果不进行归一化处理，会导致loss值不收敛，因为loss值太大了，导致梯度爆炸，所以需要进行归一化处理
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming single-channel grayscale images    
        ])

        self.setup()

    def setup(self, stage=None):
        # Create datasets with transformations
        dataset = EEG_Dataset(self.eeg_file_names, self.test_mode, transform=self.transforms)
        # Calculate the number of samples for each split
        total_size = len(dataset)
        train_size = int(self.train_size * total_size)
        val_size = int(self.val_size * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )


    # def setup(self, stage=None):
    #     # Create datasets
    #     dataset = EEG_Dataset(self.eeg_file_names, self.test_mode)
    #     self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [self.train_size, self.val_size, self.test_size], generator=torch.Generator().manual_seed(42))

    def build_dataloader(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=4)

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.build_dataloader(self.test_dataset, shuffle=False)