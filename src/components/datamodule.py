import numpy as np
import pandas as pd
import librosa
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from transformers import WhisperFeatureExtractor
import traceback

################################################################################
# Basic
################################################################################
class BirdClefDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.filepaths = self._read_filepaths(df)
        self.feature_extractor = self._create_feature_extractor()
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = self._read_labels(df[self.config["label"]])
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        sound = self._read_sound(self.filepaths[idx])
        if self.transform is not None:
            sound = self.transform(sound)
        feature = self._extract_feature(sound)
        if self.labels is not None:
            labels = self.labels[idx]
            return feature, labels
        return feature

    def _read_filepaths(self, df):
        values = df["filepath"].values
        return values

    def _create_feature_extractor(self):
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config["base_model_name"]
        )
        return feature_extractor

    def _read_sound(self, filepath):
        sound_org, _ = librosa.load(
            filepath, sr=self.config["sampling_rate"]["org"]
        )
        sound = librosa.resample(
            sound_org,
            orig_sr=self.config["sampling_rate"]["org"],
            target_sr=self.config["sampling_rate"]["target"],
        )
        return sound

    def _extract_feature(self, sound):
        feature = self.feature_extractor(
            sound,
            sampling_rate=self.config["sampling_rate"]["target"],
            return_tensors="pt"
        )["input_features"].to(torch.float32)
        return feature

    def _read_labels(self, df):
        labels = torch.tensor(df.apply(self._to_onehot), dtype=torch.float32)
        return labels

    def _to_onehot(self, series):
        return [1 if l in series else 0 for l in self.config["labels"]]


class BirdClefPredDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.filepaths = self._read_filepaths(df)
        self.end_sec = self._read_end_sec(df)
        self.feature_extractor = self._create_feature_extractor()
        self.transform = transform

        # variables
        self.wholesound = None

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        chunk = self._read_chunk(self.filepaths[idx], self.end_sec[idx])
        if self.transform is not None:
            chunk = self.transform(chunk)
        feature = self._extract_feature(chunk)
        return feature

    def _read_filepaths(self, df):
        values = self.config["path"]["preddata"] \
            + df["row_id"].str.replace("_[0-9]+$", "", regex=True) \
            + ".ogg"
        return values

    def _read_end_sec(self, df):
        end_sec = df["row_id"].apply(lambda x: x.split("_")[-1]).astype(int)
        return end_sec

    def _create_feature_extractor(self):
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config["base_model_name"]
        )
        return feature_extractor

    def _read_chunk(self, filepath, end_sec):
        if self._should_load(end_sec):
            self.wholesound = self._read_sound(filepath)
        start_sec = end_sec - self.config["chunk_sec"]
        idx_start = start_sec * self.config["sampling_rate"]["target"]
        idx_end = (start_sec + self.config["duration_sec"]) * self.config["sampling_rate"]["target"]
        chunk = self.wholesound[idx_start:idx_end]
        return chunk

    def _should_load(self, end_sec):
        return (end_sec - self.config["chunk_sec"]) == 0

    def _read_sound(self, filepath):
        wholesound_org, _ = librosa.load(
            filepath,
            sr=self.config["sampling_rate"]["org"]
        )
        wholesound = librosa.resample(
            wholesound_org,
            orig_sr=self.config["sampling_rate"]["org"],
            target_sr=self.config["sampling_rate"]["target"]
        )
        return wholesound

    def _extract_feature(self, chunk):
        feature = self.feature_extractor(
            chunk,
            sampling_rate=self.config["sampling_rate"]["target"],
            return_tensors="pt"
        )["input_features"].to(torch.float32)
        return feature


class DataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_pred, Dataset, config, transforms):
        super().__init__()

        # const
        self.config = config
        self.df_train = df_train
        self.df_val = df_val
        self.df_pred = df_pred
        self.transforms = self._read_transforms(transforms)

        # class
        self.Dataset = Dataset

    def _read_transforms(self, transforms):
        if transforms is not None:
            return transforms
        return {"train": None, "valid": None, "pred": None}

    def train_dataloader(self):
        dataset = self.Dataset(
            self.df_train, self.config["dataset"], self.transforms["train"]
        )
        return DataLoader(dataset, **self.config["train_loader"])

    def val_dataloader(self):
        dataset = self.Dataset(
            self.df_val, self.config["dataset"], self.transforms["valid"]
        )
        return DataLoader(dataset, **self.config["val_loader"])

    def predict_dataloader(self):
        dataset = self.Dataset(
            self.df_pred, self.config["dataset"], self.transforms["pred"]
        )
        return DataLoader(dataset, **self.config["pred_loader"])
