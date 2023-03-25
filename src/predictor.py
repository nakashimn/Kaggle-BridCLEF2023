import os
import pathlib
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import librosa
import pytorch_lightning as pl
from pytorch_lightning.accelerators import CPUAccelerator, CUDAAccelerator
import traceback

from components.preprocessor import DataPreprocessor
from components.datamodule import BirdClefPredDataset, DataModule
from components.models import BirdClefModel
from config.trial_v0 import config

class Predictor:
    def __init__(
        self, Model, DataModule, Dataset, df_pred, config, transforms
    ):
        # const
        self.config = config
        self.df_pred = df_pred
        self.transforms = transforms

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Dataset = Dataset

        # variables
        self.probs = None

    def run(self):
        # create datamodule
        datamodule = self._create_datamodule()

        # predict
        self.probs = self._predict(datamodule)

        return self.probs

    def _create_datamodule(self):
        datamodule = self.DataModule(
            df_train=None,
            df_val=None,
            df_pred=self.df_pred,
            Dataset=self.Dataset,
            config=self.config["datamodule"],
            transforms=self.transforms
        )
        return datamodule

    def _predict(self, datamodule):
        # define trainer
        trainer = pl.Trainer(
            logger=None,
            **self.config["trainer"]
        )

        # load model
        model = self.Model.load_from_checkpoint(
            f"{self.config['path']['model_dir']}/{self.config['modelname']}.ckpt",
            config=self.config["model"],
            transforms=self.transforms,
            map_location=torch.device(self.config["pred_device"])
        )

        # prediction
        model.eval()
        with torch.inference_mode():
            preds = trainer.predict(model, datamodule=datamodule)
        probs = np.concatenate([p["prob"].numpy() for p in preds], axis=0)
        return probs

class PredictorEnsemble(Predictor):
    def _predict(self, datamodule):
        # define trainer
        trainer = pl.Trainer(
            logger=None,
            **self.config["trainer"]
        )

        probs_folds = []
        for fold in range(self.config["n_splits"]):

            # load model
            model = self.Model.load_from_checkpoint(
                f"{self.config['path']['model_dir']}/{self.config['modelname']}_{fold}.ckpt",
                config=self.config["model"],
                transforms=self.transforms,
                map_location=torch.device(self.config["pred_device"])
            )

            # prediction
            model.eval()
            with torch.inference_mode():
                preds = trainer.predict(model, datamodule=datamodule)
            probs = np.concatenate([p["prob"].numpy() for p in preds], axis=0)
            probs_folds.append(probs)
        probs_ensemble = np.mean(probs_folds, axis=0)
        return probs_ensemble

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fix_config_for_pred(config):
    if config["pred_device"] == "cpu":
        config["trainer"]["accelerator"] = "cpu"
        config["model"]["device"] = "cpu"
    if config["pred_device"] == "gpu":
        config["trainer"]["accelerator"] = "gpu"
        config["model"]["device"] = "cuda"
    return config

if __name__=="__main__":

    fix_seed(config["random_seed"])

    # fix config
    config = fix_config_for_pred(config)

    # Setting Dataset
    data_preprocessor = DataPreprocessor(config)
    df_pred = data_preprocessor.pred_dataset_for_submit()

    # Prediction
    if config["pred_ensemble"]:
        cls_predictor = PredictorEnsemble
    else:
        cls_predictor = Predictor
    predictor = cls_predictor(
        BirdClefModel,
        DataModule,
        BirdClefPredDataset,
        df_pred,
        config,
        None
    )
    probs = predictor.run()

    # output
    submission = pd.concat([
        df_pred,
        pd.DataFrame(probs, columns=config["labels"]).drop("none", axis=1)
    ], axis=1)
    submission.to_csv("submission.csv", index=None)
