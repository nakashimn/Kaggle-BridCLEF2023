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
import traceback

from components.preprocessor import DataPreprocessor
from components.datamodule import BirdClefDataset, DataModule
from components.models import BirdClefModel
from components.validations import MinLoss, ValidResult, ConfusionMatrix, F1Score, LogLoss

class Trainer:
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

        # variable
        self.min_loss = MinLoss()
        self.val_probs = ValidResult()
        self.val_labels = ValidResult()

    def run(self):
        iterator_kfold = self._create_kfold_iterator(self.df_train)
        for fold, (idx_train, idx_val) in iterator_kfold:
            # create datamodule
            datamodule = self._create_datamodule(idx_train, idx_val)

            # train crossvalid models
            if fold in self.config["train_fold"]:
                min_loss = self._train_with_crossvalid(datamodule, fold)
                self.min_loss.update(min_loss)

            # valid
            if fold in self.config["valid_fold"]:
                val_probs, val_labels = self._valid(datamodule, fold)
                self.val_probs.append(val_probs)
                self.val_labels.append(val_labels)

            # log
        self.mlflow_logger.log_metrics({"train_min_loss": self.min_loss.value})

        # train final model
        if self.config["train_with_alldata"] or (iterator_kfold is None):
            datamodule = self._create_datamodule_with_alldata()
            self._train_without_valid(datamodule, self.min_loss.value)

    def _create_kfold_iterator(self, *df):
        if self.config["kfold"]["params"]["n_splits"] <= 1:
            return None
        class_kfold = getattr(
            sklearn.model_selection,
            self.config["kfold"]["name"]
        )
        kfold = class_kfold(**self.config["kfold"]["params"])
        iterator = enumerate(kfold.split(*df, **self.config["kfold"]["anchor"]))
        return iterator

    def _create_datamodule(self, idx_train, idx_val):
        df_train_fold = self.df_train.loc[idx_train].reset_index(drop=True)
        df_val_fold = self.df_train.loc[idx_val].reset_index(drop=True)
        datamodule = self.DataModule(
            df_train=df_train_fold,
            df_val=df_val_fold,
            df_pred=None,
            Dataset=self.Dataset,
            config=self.config["datamodule"],
            transforms=self.transforms
        )
        return datamodule

    def _create_datamodule_with_alldata(self):
        df_val_dummy = self.df_train.iloc[:10]
        datamodule = self.DataModule(
            df_train=self.df_train,
            df_val=df_val_dummy,
            df_pred=None,
            Dataset=self.Dataset,
            config=self.config["datamodule"],
            transforms=self.transforms
        )
        return datamodule

    def _train_with_crossvalid(self, datamodule, fold):
        model = self.Model(self.config["model"])
        checkpoint_name = f"best_loss_{fold}"

        earystopping = EarlyStopping(
            monitor="val_loss",
            **self.config["earlystopping"]
        )
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename=checkpoint_name,
            **self.config["checkpoint"]
        )

        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **self.config["trainer"],
        )

        trainer.fit(model, datamodule=datamodule)

        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
            f"{self.config['path']['temporal_dir']}/{checkpoint_name}.ckpt"
        )

        min_loss = copy.deepcopy(model.min_loss)

        del model
        gc.collect()

        return min_loss

    def _train_without_valid(self, datamodule, min_loss):
        model = self.Model(self.config["model"])
        checkpoint_name = f"best_loss"

        earystopping = EarlyStopping(
            monitor="train_loss",
            stopping_threshold=min_loss,
            **self.config["earlystopping"]
        )
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename=checkpoint_name,
            **self.config["checkpoint"]
        )

        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **self.config["trainer"],
        )

        trainer.fit(model, datamodule=datamodule)

        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
            f"{self.config['path']['temporal_dir']}/{checkpoint_name}.ckpt"
        )

        del model
        gc.collect()

    def _valid(self, datamodule, fold):
        checkpoint_name = f"best_loss_{fold}"
        model = self.Model.load_from_checkpoint(
            f"{self.config['path']['temporal_dir']}/{checkpoint_name}.ckpt",
            config=self.config["model"]
        )
        model.eval()

        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            **self.config["trainer"]
        )

        trainer.validate(model, datamodule=datamodule)

        val_probs = copy.deepcopy(model.val_probs)
        val_labels = copy.deepcopy(model.val_labels)

        del model
        gc.collect()

        return val_probs, val_labels

def create_mlflow_logger(config):
    timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=timestamp
    )
    return mlflow_logger

def update_model(config, filepath_config):
    filepaths_ckpt = glob.glob(f"{config['path']['temporal_dir']}/*.ckpt")
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    for filepath_ckpt in filepaths_ckpt:
        filename = pathlib.Path(filepath_ckpt).name
        shutil.move(filepath_ckpt, str(dirpath_model / filename))
    filename_config = pathlib.Path(filepath_config).name
    shutil.copy2(filepath_config, str(dirpath_model / filename_config))

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="stem of config filepath.",
        type=str,
        required=True
    )
    parser.add_argument(
        "-m", "--message",
        help="message for upload to kaggle datasets.",
        type=str,
        required=False
    )
    return parser.parse_args()


if __name__=="__main__":

    # args
    args = get_args()
    config = importlib.import_module(f"config.{args.config}").config

    # logger
    mlflow_logger = create_mlflow_logger(config)
    mlflow_logger.log_hyperparams(config)
    mlflow_logger.experiment.log_artifact(
        mlflow_logger.run_id,
        f"./config/{args.config}.py"
    )

    # torch setting
    torch.set_float32_matmul_precision("medium")

    # Preprocess
    data_preprocessor = DataPreprocessor(config)
    fix_seed(config["random_seed"])
    df_train = data_preprocessor.train_dataset()
    # df_train = data_preprocessor.train_dataset_primary()

    # Training
    trainer = Trainer(
        BirdClefModel,
        DataModule,
        BirdClefDataset,
        df_train,
        config,
        None,
        mlflow_logger
    )
    trainer.run()

    # Validation Result

    # # Update model
    update_model(config, f"./config/{args.config}.py")
    if args.message is not None:
        upload_model(config, args.message)
