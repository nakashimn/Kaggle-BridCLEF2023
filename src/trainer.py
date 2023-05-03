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
from components.augmentation import SoundAugmentation
from components.models import BirdClefModel
from components.validations import MinLoss, ValidResult, ConfusionMatrix, CMeanAveragePrecision

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

        idx_train, idx_val = self._split_dataset(self.df_train)
        # create datamodule
        datamodule = self._create_datamodule(idx_train, idx_val)

        # train crossvalid models
        min_loss = self._train_with_crossvalid(datamodule, fold)
        self.min_loss.update(min_loss)

        # valid
        val_probs, val_labels = self._valid(datamodule, fold)
        self.val_probs.append(val_probs)
        self.val_labels.append(val_labels)

        # log
        self.mlflow_logger.log_metrics({"train_min_loss": self.min_loss.value})

        # train final model
        if self.config["train_with_alldata"]:
            datamodule = self._create_datamodule_with_alldata()
            self._train_without_valid(datamodule, self.min_loss.value)

    def _split_dataset(self, df):
        # pickup minor label
        df_start = df.loc[df["filename"].str.contains("_5.npz")]
        counts = df_start["primary_label"].value_counts()
        minor_labels = list(counts[counts<2].index)

        # get training indices
        idx_train = df.loc[
            ~df["filename"].str.contains("_5.npz") |
            df["primary_label"].isin(minor_labels)
        ].index

        # get validation indices
        idx_val = df.loc[
            df["filename"].str.contains("_5.npz") &
            ~df["primary_label"].isin(minor_labels)
        ].index
    return idx_train, idx_val

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
    # df_train = data_preprocessor.train_dataset()
    df_train = data_preprocessor.train_dataset_primary()

    # Augmentation
    sound_augmentation = None
    if config["augmentation"] is not None:
        sound_augmentation = SoundAugmentation(
            **config["augmentation"]
        )
    transforms = {
        "train": sound_augmentation,
        "valid": None,
        "pred": None
    }

    # Training
    trainer = Trainer(
        BirdClefModel,
        DataModule,
        BirdClefDataset,
        df_train,
        config,
        transforms,
        mlflow_logger
    )
    trainer.run()

    # Validation Result
    # confmat = ConfusionMatrix(
    #     trainer.val_probs.values,
    #     trainer.val_labels.values,
    #     config["Metrics"]["confmat"]
    # )
    # fig_confmat = confmat.draw()
    # fig_confmat.savefig(f"{config['path']['temporal_dir']}/confmat.png")
    # mlflow_logger.experiment.log_artifact(
    #     mlflow_logger.run_id,
    #     f"{config['path']['temporal_dir']}/confmat.png"
    # )
    cmap = CMeanAveragePrecision(
        trainer.val_probs.values,
        trainer.val_labels.values,
        config["Metrics"]["cmAP"]
    ).calc()
    mlflow_logger.log_metrics({
        "cmAP": cmap
    })
    print(f"cmAP:{cmap:.03f}")

    # # Update model
    update_model(config, f"./config/{args.config}.py")
    if args.message is not None:
        upload_model(config, args.message)
