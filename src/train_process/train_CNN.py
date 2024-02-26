import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from distutils.util import strtobool
import argparse
import logging

import sys
sys.path.append('../src/data')
from data.data_module import EEG_DataModule
from model.CNN import CNN


logging.disable(logging.CRITICAL)
RANDOM_SEED = 42
pl.seed_everything(seed=RANDOM_SEED)



def train(train_args):
    """
    Use Pytorch Lightning to train the CNN model.
    """
    pl.seed_everything(seed=train_args.get("seed", 42), workers=True)
    # Create a folder to save the checkpoints of the CNN model if it's not exist
    CNN_cpt = train_args.get("out")
    if not os.path.exists(CNN_cpt): os.makedirs(CNN_cpt)

    # Prepare Data Module
    data_module = EEG_DataModule(
        eeg_file_names = train_args.get("eeg_file_names"),
        test_mode = train_args.get("test_mode"),
        test_on_each_patient = train_args.get("test_on_each_patient"),
        image_size = train_args.get("image_size"),
        batch_size = train_args.get("batch_size"),
        train_size = train_args.get("train_size"),
        val_size = train_args.get("val_size"),
        test_size = train_args.get("tests_size")
    )


    # Prepare Model
    model = CNN(learning_rate = train_args.get("learning_rate"))

    # Initialize Wandb logger: `wandb`实例可以在训练过程中记录模型的参数和指标，以便在W&B仪表板中查看，并直接传递给PL的`Trainer`实例
    if not train_args.get("disable_wandb", False):
        wandb_logger = WandbLogger(project='EEG_classification_with_CNN', 
                                   name=f"{ train_args.get('wandb_run_name', 'None') }")


    # Initialize Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss") 

    checkpoints_callback = ModelCheckpoint(
                                save_top_k=1,
                                monitor="train_loss",
                                mode="min",
                                dirpath=train_args.get("out"),
                                filename=train_args.get('wandb_run_name'),
                                every_n_epochs=1)
    
    callbacks = [early_stop_callback, checkpoints_callback] 

    # Define the pl trainer
    trainer = pl.Trainer(   max_epochs=train_args.get("max_epochs"),
                            accelerator="gpu", 
                            devices=1,
                            logger=wandb_logger, 
                            enable_progress_bar=train_args.get("enable_progress_bar"),
                            precision='16-mixed',
                            callbacks=callbacks)
    # Train the model
    trainer.fit(model=model, datamodule=data_module)

    # Validate the model 
    trainer.validate(model=model, datamodule=data_module)

    # Test the model
    if not train_args.get("test_on_each_patient"):
        trainer.test(model=model, datamodule=data_module)
    else:
        patient_test_dataloaders = data_module.get_patient_test_dataloaders()
        # 遍历每个患者的DataLoader进行测试
        for i, dataloader in enumerate(patient_test_dataloaders):
            print(f"Testing 实验对象 {i + 1}/{len(patient_test_dataloaders)}")
            # 在这里，每次调用test只测试一个患者的数据
            trainer.test(model=model, dataloaders=dataloader)

