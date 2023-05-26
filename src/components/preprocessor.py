import os
import glob
import pathlib
import ast
from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd
import traceback

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def train_dataset(self):
        df_meta = pd.read_csv(self.config["path"]["trainmeta"])
        df_meta = self._cleansing(df_meta)
        df = pd.DataFrame(
            (self.config["path"]["traindata"] + df_meta["filename"]).values,
            columns=["filepath"]
        )
        df["labels"] = df_meta.apply(self._aggregate_labels, axis=1)
        return df

    def train_dataset_primary(self):
        df_meta = pd.read_csv(self.config["path"]["trainmeta"])
        df_meta = self._cleansing(df_meta)
        df = pd.DataFrame(
            (self.config["path"]["traindata"] + df_meta["filename"]).values,
            columns=["filepath"]
        )
        df["labels"] = df_meta.apply(self._primary_label, axis=1)
        df["label_id"] = self._label_to_id(df)
        return df

    def _label_to_id(self, df):
        labels = self.config["labels"]
        return [labels.index(l[0]) for l in df["labels"]]

    def train_dataset_for_pretrain(self):
        df_meta = pd.read_csv(self.config["path"]["trainmeta"])
        return df_meta

    def train_dataset_for_bg_classifier(self):
        df_meta = pd.read_csv(self.config["path"]["trainmeta"])
        df_meta = self._cleansing(df_meta)
        df = pd.DataFrame(
            (self.config["path"]["traindata"] + df_meta["filename"]).values,
            columns=["filepath"]
        )
        df["labels"] = df_meta.apply(self._activity, axis=1)
        return df

    def test_dataset(self):
        df_meta = pd.read_csv(self.config["path"]["trainmeta"])
        df = pd.DataFrame(
            (self.config["path"]["traindata"] + df_meta["filename"]).values,
            columns=["filepath"]
        )
        df["labels"] = df_meta.apply(self._aggregate_labels, axis=1)
        return df

    def test_dataset_primary(self):
        df_meta = pd.read_csv(self.config["path"]["trainmeta"])
        df = pd.DataFrame(
            (self.config["path"]["traindata"] + df_meta["filename"]).values,
            columns=["filepath"]
        )
        df["labels"] = df_meta.apply(self._primary_label, axis=1)
        return df

    def pred_dataset(self):
        filepaths = glob.glob(
            f"{self.config['path']['preddata']}/**/*.npz", recursive=True
        )
        df = pd.DataFrame(filepaths, columns=["filepath"])
        return df

    def pred_dataset_for_submit(self):
        df = self.pred_dataset()
        df_pred = self._create_submit_indices(df)
        return df_pred

    def _cleansing(self, df):
        return df

    def _aggregate_labels(self, series):
        return [series["primary_label"]] + ast.literal_eval(series["secondary_labels"])

    def _primary_label(self, series):
        return [series["primary_label"]]

    def _activity(self, series):
        return [str(series["activity"])]

    def _create_submit_indices(self, df):
        dict_chunk = {
            "row_id": [],
        }
        for idx in tqdm(df.index):
            filepath = df.loc[idx, "filepath"]
            filestem = pathlib.Path(filepath).stem

            snd_sec = librosa.get_duration(
                filename=filepath, sr=self.config["sampling_rate"]
            )
            num_chunk = int(np.ceil(snd_sec / self.config["chunk_sec"]))

            dict_chunk["row_id"] += [
                f"{filestem}_{int(self.config['chunk_sec']*i)}"
                for i in range(1, num_chunk+1)
            ]
        df_pred = pd.DataFrame(dict_chunk)
        return df_pred
