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
from components.datamodule import BirdClefMelspecDataset, DataModule
from components.augmentation import SpecAugmentation
from components.models import BirdClefTimmSEDModel
from components.validations import MinLoss, ValidResult, ConfusionMatrix, CMeanAveragePrecision

class ModelUploader(callbacks.Callback):
    def __init__(self, model_dir, message=""):
        self.model_dir = model_dir
        self.message = message
        super().__init__()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        epoch = trainer.current_epoch
        if (epoch % 5 == 0):
            self._upload_model(f"{self.message}[epoch_{epoch}]")
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def _upload_model(self, message):
        try:
            subprocess.run(
                ["kaggle", "datasets", "version", "-m", message],
                cwd=self.model_dir
            )
        except:
            print(traceback.format_exc())

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

        # idx_train, idx_val = self._split_dataset(self.df_train)
        kfold = sklearn.model_selection.StratifiedKFold(
            n_splits=5, shuffle=True, random_state=config["random_seed"]
        )
        for idx_train, idx_val in kfold.split(self.df_train, self.df_train["label_id"]):
            break

        # create datamodule
        datamodule = self._create_datamodule(idx_train, idx_val)

        # train crossvalid models
        min_loss = self._train_with_crossvalid(datamodule, 0)
        self.min_loss.update(min_loss)

        # valid
        val_probs, val_labels = self._valid(datamodule, 0)
        self.val_probs.append(val_probs)
        self.val_labels.append(val_labels)

        # log
        self.mlflow_logger.log_metrics({"train_min_loss": self.min_loss.value})

        # train final model
        if self.config["train_with_alldata"]:
            datamodule = self._create_datamodule_with_alldata()
            self._train_without_valid(datamodule, self.min_loss.value)

    def _split_dataset(self, df, duration=5, ext="npz", th_counts=1):
        anchor_word = f"_{duration:d}.{ext}"
        # pickup minor label
        df_start = df.loc[df["filepath"].str.contains(anchor_word)]
        counts = df_start["labels"].value_counts()
        minor_labels = list(counts[counts<=th_counts].index)

        # get training indices
        idx_train = df.loc[
            ~df["filepath"].str.contains(anchor_word) |
            df["labels"].isin(minor_labels)
        ].index

        # get validation indices
        idx_val = df.loc[
            df["filepath"].str.contains(anchor_word) &
            ~df["labels"].isin(minor_labels)
        ].index
        print(f"train:{len(idx_train)} / val:{len(idx_val)}")
        return idx_train, idx_val

    def _create_datamodule(self, idx_train, idx_val):
        # fold dataset
        df_train_fold = self.df_train.loc[idx_train].reset_index(drop=True)
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

    def _create_datamodule_with_alldata(self):
        # create dummy data for validation
        df_val_dummy = self.df_train.iloc[:10]
        # create datamodule
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
        # create model
        model = self.Model(self.config["model"])

        ###
        # define pytorch_lightning callbacks
        ###
        # define earlystopping
        earlystopping = EarlyStopping(
            monitor="val_loss",
            **self.config["earlystopping"]
        )
        # define learning rate monitor
        lr_monitor = callbacks.LearningRateMonitor()
        # define check point
        checkpoint_name = f"{self.config['modelname']}_{fold}"
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename=checkpoint_name,
            monitor="val_loss",
            **self.config["checkpoint"]
        )
        # define model uploader
        model_uploader = ModelUploader(
            model_dir=self.config["path"]["model_dir"]
        )

        # define trainer
        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            callbacks=[lr_monitor, loss_checkpoint, earlystopping, model_uploader],
            **self.config["trainer"],
        )

        # train
        filepath_checkpoint = None
        if self.config["use_checkpoint"] and os.path.exists(self.config["path"]["checkpoint"]):
            filepath_checkpoint = self.config["path"]["checkpoint"]
        trainer.fit(model, datamodule=datamodule, ckpt_path=filepath_checkpoint)

        # logging
        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
            f"{self.config['path']['model_dir']}/{checkpoint_name}.ckpt"
        )
        min_loss = copy.deepcopy(model.min_loss)

        # clean memory
        del model
        gc.collect()

        return min_loss

    def _train_without_valid(self, datamodule, min_loss):
        # create model
        model = self.Model(self.config["model"])

        ###
        # define pytorch_lightning callbacks
        ###
        # define earlystopping
        earlystopping = EarlyStopping(
            monitor="train_loss",
            stopping_threshold=min_loss,
            **self.config["earlystopping"]
        )
        # define learning rate monitor
        lr_monitor = callbacks.LearningRateMonitor()
        # define check point
        checkpoint_name = self.config["modelname"]
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename=checkpoint_name,
            monitor="train_loss",
            **self.config["checkpoint"]
        )
        # define model uploader
        model_uploader = ModelUploader(
            model_dir=self.config["path"]["model_dir"]
        )

        # define trainer
        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            callbacks=[lr_monitor, loss_checkpoint, earlystopping, model_uploader],
            **self.config["trainer"],
        )

        # train
        filepath_checkpoint = None
        if self.config["use_checkpoint"] and os.path.exists(self.config["path"]["checkpoint"]):
            filepath_checkpoint = self.config["path"]["checkpoint"]
        trainer.fit(model, datamodule=datamodule, ckpt_path=filepath_checkpoint)

        # logging
        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
            f"{self.config['path']['model_dir']}/{checkpoint_name}.ckpt"
        )

        # clean memory
        del model
        gc.collect()

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
        trainer.validate(model, datamodule=datamodule, filepath_checkpoint=filepath_checkpoint)

        # get result
        val_probs = copy.deepcopy(model.val_probs)
        val_labels = copy.deepcopy(model.val_labels)

        # clean memory
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

def update_config(config, filepath_config):
    # copy ConfigFile from temporal_dir to model_dir
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    filename_config = pathlib.Path(filepath_config).name
    shutil.copy2(filepath_config, str(dirpath_model / filename_config))

def remove_exist_models(config):
    filepaths_ckpt = glob.glob(f"{config['path']['model_dir']}/*.ckpt")
    for fp in filepaths_ckpt:
        os.remove(fp)

def update_model(config):
    # copy Models from temporal_dir to model_dir
    filepaths_ckpt = glob.glob(f"{config['path']['temporal_dir']}/*.ckpt")
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    for filepath_ckpt in filepaths_ckpt:
        filename = pathlib.Path(filepath_ckpt).name
        shutil.move(filepath_ckpt, str(dirpath_model / filename))

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

    # Update config
    update_config(config, f"./config/{args.config}.py")

    # torch setting
    torch.set_float32_matmul_precision("medium")

    # Preprocess
    remove_exist_models(config)
    fix_seed(config["random_seed"])
    data_preprocessor = DataPreprocessor(config)
    df_train = data_preprocessor.train_dataset_primary()

    # Augmentation
    spec_augmentation = None
    if config["augmentation"] is not None:
        spec_augmentation = SpecAugmentation(
            config["augmentation"]
        )
    transforms = {
        "train": spec_augmentation,
        "valid": None,
        "pred": None
    }

    # Training
    trainer = Trainer(
        BirdClefTimmSEDModel,
        DataModule,
        BirdClefMelspecDataset,
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

    # Update model
    update_model(config)
    if args.message is not None:
        upload_model(config, args.message)
