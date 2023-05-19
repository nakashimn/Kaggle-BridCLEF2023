import os
import shutil
import copy
import importlib
import argparse
import random
import gc
import subprocess
import pathlib
import glob
import datetime
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import sklearn.model_selection
import mlflow
import traceback

from components.preprocessor import DataPreprocessor
from components.validations import MinLoss, ValidResult, ConfusionMatrix, F1Score, LogLoss

class ModelUploader(callbacks.Callback):
    def __init__(self, model_dir, every_n_epochs=5, message=""):
        self.model_dir = model_dir
        self.every_n_epochs = every_n_epochs
        self.message = message
        self.should_upload = False
        super().__init__()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if self.should_upload:
            self._upload_model(
                f"{self.message}[epoch:{trainer.current_epoch}]"
            )
            self.should_upload = False
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self.every_n_epochs is None:
            return super().on_train_epoch_end(trainer, pl_module)
        if (self.every_n_epochs <= 0):
            return super().on_train_epoch_end(trainer, pl_module)
        if (trainer.current_epoch % self.every_n_epochs == 0):
            self.should_upload = True
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs is not None:
            self._upload_model(
                f"{self.message}[epoch:{trainer.current_epoch}]"
            )
        return super().on_train_end(trainer, pl_module)

    def _upload_model(self, message):
        try:
            subprocess.run(
                ["kaggle", "datasets", "version", "-m", message],
                cwd=self.model_dir
            )
        except:
            print(traceback.format_exc())

class TrainerForPretrain:
    def __init__(
        self, Model, DataModule, Dataset,
        df_train, config, transforms, mlflow_logger
    ):
        # const
        self.mlflow_logger = mlflow_logger
        self.config = config
        self.df_train = df_train
        self.transforms = transforms

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Dataset = Dataset

    def run(self):
        # train
        datamodule = self._create_datamodule()
        self._train(datamodule)

    def _create_datamodule(self, idx_train=None, idx_val=None):
        # fold dataset
        if (idx_train is None):
            df_train_fold = self.df_train
        else:
            df_train_fold = self.df_train.loc[idx_train].reset_index(drop=True)
        if (idx_val is None):
            df_val_fold = None
        else:
            df_val_fold = self.df_train.loc[idx_val].reset_index(drop=True)

        # create datamodule
        datamodule = self.DataModule(
            df_train=df_train_fold,
            df_val=df_val_fold,
            df_pred=None,
            Dataset=self.Dataset,
            config=self.config["datamodule"],
            transforms=self.transforms
        )
        return datamodule

    def _create_model(self, filepath_checkpoint=None):
        if filepath_checkpoint is None:
            return self.Model(self.config["model"])
        if not self.config["init_with_checkpoint"]:
            return self.Model(self.config["model"])
        if not os.path.exists(filepath_checkpoint):
            return self.Model(self.config["model"])
        model = self.Model.load_from_checkpoint(
            filepath_checkpoint,
            config=self.config["model"]
        )
        return model

    def _define_monitor_value(self, fold=None):
        return "train_loss" if (fold is None) else "val_loss"

    def _define_checkpoint_name(self, fold=None):
        checkpoint_name = f"{self.config['modelname']}"
        if fold is None:
            return checkpoint_name
        checkpoint_name += f"_{fold}"
        return checkpoint_name

    def _define_checkpoint_path(self, checkpoint_name):
        filepath_ckpt_load = f"{self.config['path']['ckpt_dir']}/{checkpoint_name}.ckpt"
        filepath_ckpt_save = f"{self.config['path']['model_dir']}/{checkpoint_name}.ckpt"
        filepath_ckpt = {
            "init": None,
            "resume": None,
            "save": filepath_ckpt_save
        }
        if not os.path.exists(filepath_ckpt_load):
            return filepath_ckpt
        if self.config["resume"]:
            filepath_ckpt["resume"] = filepath_ckpt_load
        if self.config["init_with_checkpoint"]:
            filepath_ckpt["init"] = filepath_ckpt_load
        return filepath_ckpt

    def _define_callbacks(self, callback_config):
        # define earlystopping
        earlystopping = EarlyStopping(
            **callback_config["earlystopping"],
            **self.config["earlystopping"]
        )
        # define learning rate monitor
        lr_monitor = callbacks.LearningRateMonitor()
        # define check point
        loss_checkpoint = callbacks.ModelCheckpoint(
            **callback_config["checkpoint"],
            **self.config["checkpoint"]
        )
        # define model uploader
        model_uploader = ModelUploader(
            model_dir=self.config["path"]["model_dir"],
            every_n_epochs=self.config["upload_every_n_epochs"],
            message=self.config["experiment_name"]
        )

        callback_list = [
            earlystopping, lr_monitor, loss_checkpoint, model_uploader
        ]
        return callback_list

    def _train(self, datamodule, fold=None, min_delta=0.0, min_loss=None):
        # switch mode
        monitor = self._define_monitor_value(fold)

        # define saved checkpoint name
        checkpoint_name = self._define_checkpoint_name(fold)

        # define loading checkpoint
        filepath_checkpoint = self._define_checkpoint_path(checkpoint_name)

        # define pytorch_lightning callbacks
        callback_config = {
            "earlystopping": {
                "monitor": monitor,
                "min_delta": min_delta,
                "stopping_threshold": min_loss
            },
            "checkpoint": {
                "filename": checkpoint_name,
                "monitor": monitor
            }
        }
        callback_list = self._define_callbacks(callback_config)

        # create model
        model = self._create_model(filepath_checkpoint["init"])

        # define trainer
        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            callbacks=callback_list,
            fast_dev_run=False,
            num_sanity_val_steps=0,
            **self.config["trainer"]
        )

        # train
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=filepath_checkpoint["resume"]
        )

        # logging
        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
            filepath_checkpoint["save"]
        )
        min_loss = copy.deepcopy(model.min_loss)

        # clean memory
        del model
        gc.collect()

        return min_loss

    def _valid(self, datamodule, fold):
        # load model
        model = self.Model(self.config["model"])
        model.eval()

        # define trainer
        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            **self.config["trainer"]
        )

        # validation
        filepath_checkpoint = f"{self.config['path']['model_dir']}/{self.config['modelname']}_{fold}.ckpt"
        trainer.validate(
            model,
            datamodule=datamodule,
            ckpt_path=filepath_checkpoint
        )

        # get result
        val_probs = copy.deepcopy(model.val_probs)
        val_labels = copy.deepcopy(model.val_labels)

        # clean memory
        del model
        gc.collect()

        return val_probs, val_labels

def create_mlflow_logger(config):
    # create Logger instance
    timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=timestamp,
        run_id=config["run_id"]
    )
    # debug message
    print("MLflow:")
    print(f"  experiment_name: {config['experiment_name']}")
    print(f"  run_name: {timestamp}")
    print(f"  run_id: {mlflow_logger.run_id}")
    return mlflow_logger

def update_config(config, filepath_config):
    # copy ConfigFile from temporal_dir to model_dir
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    filename_config = pathlib.Path(filepath_config).name
    shutil.copy2(filepath_config, str(dirpath_model / filename_config))

def remove_exist_models(config):
    filepaths_ckpt = glob.glob(f"{config['path']['model_dir']}/*.ckpt")
    for fp in filepaths_ckpt:
        os.remove(fp)

def upload_model(config, message):
    try:
        subprocess.run(
            ["kaggle", "datasets", "version", "-m", message],
            cwd=config["path"]["model_dir"]
        )
    except:
        print(traceback.format_exc())

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_classes(config):
    # import Classes dynamically
    Model = getattr(
        importlib.import_module(f"components.models"),
        config["model"]["ClassName"]
    )
    Dataset = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["dataset"]["ClassName"]
    )
    DataModule = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["ClassName"]
    )
    Augmentation = getattr(
        importlib.import_module(f"components.augmentation"),
        config["augmentation"]["ClassName"]
    )
    # debug message
    print(f"Model: {Model.__name__}")
    print(f"Dataset: {Dataset.__name__}")
    print(f"DataModule: {DataModule.__name__}")
    print(f"Augmentation: {Augmentation.__name__}")
    return Model, Dataset, DataModule, Augmentation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="stem of config filepath.",
        type=str,
        required=True
    )
    return parser.parse_args()


if __name__=="__main__":

    # args
    args = get_args()
    config = importlib.import_module(f"config.{args.config}").config

    # import Classes
    Model, Dataset, DataModule, Augmentation = import_classes(config)

    # logger
    mlflow_logger = create_mlflow_logger(config)
    mlflow_logger.log_hyperparams(config)
    mlflow_logger.experiment.log_artifact(
        mlflow_logger.run_id,
        f"./config/{args.config}.py"
    )

    # Update config
    update_config(config, f"./config/{args.config}.py")

    # torch setting
    torch.set_float32_matmul_precision("medium")

    try:
        # Preprocess
        remove_exist_models(config)
        fix_seed(config["random_seed"])
        data_preprocessor = DataPreprocessor(config)
        df_train = data_preprocessor.train_dataset_for_pretrain()

        # Augmentation
        spec_augmentation = Augmentation(
            config["augmentation"]
        )
        transforms = {
            "train": spec_augmentation,
            "valid": None,
            "pred": None
        }

        # Training
        trainer = TrainerForPretrain(
            Model,
            DataModule,
            Dataset,
            df_train,
            config,
            transforms,
            mlflow_logger
        )
        trainer.run()

    except KeyboardInterrupt:
        print(f"mlflow.run_id: {mlflow_logger.run_id}")

    except:
        print(traceback.format_exc())
        mlflow.delete_run(mlflow_logger.run_id)
