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
import random


class EEG_Dataset(Dataset):
    def __init__(self, eeg_file_names, test_mode=False, transform=None, is_single_channel=True):
        self.all_images, self.all_labels = self.prepare_data_from_multi_file(eeg_file_names, test_mode)
        self.transform = transform
        self.is_single_channel = is_single_channel

    def __len__(self) -> int:
        return len(self.all_images)

    def __getitem__(self, index: int):
        # 使用 PIL 加载图像，而不是直接使用 numpy 数组
        image_path = self.all_images[index]
        if self.is_single_channel:
            image = Image.open(image_path).convert('L')  # 转换为灰度图像
        else:
            image = Image.open(image_path).convert('RGB')

        # 应用传递给类的转换
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.all_labels[index], dtype=torch.int64)
        return image, label

    def prepare_data_from_multi_file(self, eeg_file_names, test_mode):
        """
        Load all the data into two huge lists: all_images and all_labels
        """
        all_images = []
        all_labels = []

        for eeg_file_name in eeg_file_names:
            # 读取每个图像的标签
            with pd.ExcelFile(rf'../src/data/EEG_DATASET/{eeg_file_name}/{eeg_file_name}.xlsx') as xls:
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
        # 统计一下正类和负类的数量分别是多少
        num_positive = all_labels_np.sum()
        num_negative = len(all_labels_np) - num_positive
        print(f"Jie Log: num_positive: {num_positive}, num_negative: {num_negative}")


        return all_images, all_labels_np



class EEG_DataModule(pl.LightningDataModule):

    def __init__(self,
                 eeg_file_names: List[str],
                 test_mode: bool = False,
                 test_on_each_patient: bool = True,
                 is_single_channel: bool = True,
                 image_size: Union[int, None] = None,
                 batch_size: int = 32,
                 train_size: float = 0.7, val_size: float = 0.1, test_size: float = 0.2,
                 ):
        super().__init__()
        self.eeg_file_names = eeg_file_names
        self.test_mode = test_mode
        self.test_on_each_patient = test_on_each_patient
        self.is_single_channel = is_single_channel
        
        self.image_size = image_size
        self.batch_size = batch_size

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        
        #图片的归一化处理，原因是：在训练过程中，如果不进行归一化处理，会导致loss值不收敛，因为loss值太大了，导致梯度爆炸，所以需要进行归一化处理

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size),antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming single-channel grayscale images    
        ])


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """
        This method will be automatically called by pl at the beginning of training and validation.
        """
        # Create datasets with transformations
        dataset = EEG_Dataset(self.eeg_file_names, self.test_mode, transform=self.transforms, is_single_channel=self.is_single_channel)
        # Calculate the number of samples for each split
        total_size = len(dataset)
        print(f"Jie Log: total_size: {total_size}")
        train_size = int(self.train_size * total_size)
        val_size = int(self.val_size * total_size)
        test_size = total_size - train_size - val_size
        
        # Split the dataset according to the config (test on each patient or not)
        if not self.test_on_each_patient:
            print(F"Jie Log: 使用random split对数据集进行划分, 不对每一个人进行test")
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            DATA_PER_PEOPLE = 3600
            # 划分每个人的数据
            train_data, val_data, test_data = [], [], []
            for i in range(len(self.eeg_file_names)):
                start_idx, end_idx = i * DATA_PER_PEOPLE, (i + 1) * DATA_PER_PEOPLE
                _all_idx = list(range(start_idx, end_idx))
                random.shuffle(_all_idx)
                train, val, test = _all_idx[:int(0.7 * DATA_PER_PEOPLE)], _all_idx[int(0.7 * DATA_PER_PEOPLE):int(0.8 * DATA_PER_PEOPLE)], _all_idx[int(0.8 * DATA_PER_PEOPLE):]

                # 将数据添加到总的数据集中
                train_data += train
                val_data += val
                test_data += test

            # 使用划分的索引创建子数据集
            self.train_dataset = torch.utils.data.Subset(dataset, train_data)
            self.val_dataset = torch.utils.data.Subset(dataset, val_data)
            self.test_dataset = torch.utils.data.Subset(dataset, test_data)
            
        print(f"Jie Log: EEG_DataModule.setup()执行完毕： train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")


    def build_dataloader(self, dataset, shuffle=True):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0,persistent_workers=False)

    def train_dataloader(self):
        print(f"Jie Log: EEG_DataModule.train_dataloader()开始执行") 
        train_dataloader = self.build_dataloader(self.train_dataset, shuffle=True)
        print(f"Jie Log: EEG_DataModule.train_dataloader()执行完毕")
        return train_dataloader

    def val_dataloader(self):
        return self.build_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.build_dataloader(self.test_dataset, shuffle=False)

    def get_patient_test_dataloaders(self):
        patient_test_dataloaders = []
        data_per_patient = 720  # 每个患者的测试数据量（3600*0.2）
        for i in range(len(self.eeg_file_names)):  
            start_idx = i * data_per_patient
            end_idx = start_idx + data_per_patient
            patient_test_dataset = torch.utils.data.Subset(self.test_dataset, list(range(start_idx, end_idx)))
            patient_test_dataloader = self.build_dataloader(patient_test_dataset, shuffle=False)
            patient_test_dataloaders.append(patient_test_dataloader)
        return patient_test_dataloaders
