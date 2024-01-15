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
        image_size = train_args.get("image_size"),
        batch_size = train_args.get("batch_size"),
        train_size = train_args.get("train_size"),
        val_size = train_args.get("val_size"),
        test_size = train_args.get("test_size")
    )

    # Prepare Model
    model = CNN()

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

    # Validate the model TODO: define the validate metrics
    trainer.validate(model=model, datamodule=data_module)

    # Test the model
    trainer.test(model=model, datamodule=data_module)





# If you want to trin CNN by terminal you can use the following code

# def str2bool(v):
#     return bool(strtobool(v))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train EEG Classification Model with CNN.")

#     # Arguments for train_args
#     parser.add_argument("--eeg_file_names", type=str, nargs='+', required=True, help="List of EEG file names for training.")
#     parser.add_argument("--test_mode", type=str2bool, default=False, help="Whether to run in test mode.")
#     parser.add_argument("--image_size", type=int, default=128, help="Image size for CNN input.")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
#     parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of training set.")
#     parser.add_argument("--val_size", type=float, default=0.2, help="Proportion of validation set.")
#     parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of test set.")
#     parser.add_argument("--out", type=str, required=True, help="Directory to save model checkpoints.")
#     parser.add_argument("--wandb_run_name", type=str, default='EEG_Classification', help="W&B run name.")
#     parser.add_argument("--disable_wandb", type=str2bool, default=False, help="Disable logging to W&B.")
#     parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs.")

#     args = parser.parse_args()

#     # Convert parsed arguments into a dictionary
#     train_args = vars(args)

#     # Train the model
#     train(train_args)