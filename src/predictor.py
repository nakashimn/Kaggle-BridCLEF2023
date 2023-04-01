import os
import pathlib
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import librosa
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import traceback

from components.preprocessor import DataPreprocessor
from components.datamodule import BirdClefPredDataset, DataModule
from components.models import BirdClefModel
from config.sample import config

class Predictor:
    def __init__(
        self, Model, Dataset, df_pred, config, transforms
    ):
        # const
        self.config = config
        self.df_pred = df_pred
        self.transforms = transforms

        # Class
        self.Model = Model
        self.Dataset = Dataset

        # variables
        self.probs = None

    def run(self):
        # create dataloader
        dataloader = self._create_dataloader()

        # predict
        self.probs = self._predict(dataloader)

        return self.probs

    def _create_dataloader(self):
        dataset = self.Dataset(df_pred, config["datamodule"]["dataset"])
        dataloader = DataLoader(
            dataset=dataset,
            num_workers=0,
            batch_size=4,
            shuffle=False,
            drop_last=False
        )
        return dataloader

    def _predict(self, dataloader):
        # load model
        model = self.Model.load_from_checkpoint(
            f"{self.config['path']['model_dir']}/{self.config['modelname']}.ckpt",
            config=self.config["model"],
            transforms=self.transforms,
            map_location=torch.device(self.config["pred_device"])
        )

        # prediction
        probs_batch = []
        model.eval()
        with torch.inference_mode():
            for data in tqdm(dataloader):
                preds = model.predict_step(data, None)
                probs_batch.append(preds["prob"].numpy())
        probs = np.concatenate(probs_batch, axis=0)
        return probs

class PredictorEnsemble(Predictor):
    def _predict(self, dataloader):

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
            probs_batch = []
            model.eval()
            with torch.inference_mode():
                for data in tqdm(dataloader):
                    preds = model.predict_step(data, None)
                    probs_batch.append(preds["prob"].numpy())
            probs = np.concatenate(probs_batch, axis=0)
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
