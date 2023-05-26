import numpy as np
import pandas as pd
import librosa
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as Tv
import albumentations as A
from pytorch_lightning import LightningDataModule
from transformers import Wav2Vec2FeatureExtractor
import traceback

################################################################################
# For Wav2Vec2
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
        feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=self.config["sampling_rate"]["target"])
        return feature_extractor

    def _read_sound(self, filepath):
        sound = np.load(filepath)["arr_0"]
        if (self.config["sampling_rate"]["org"]==self.config["sampling_rate"]["target"]):
            return sound
        sound = librosa.resample(
            sound,
            orig_sr=self.config["sampling_rate"]["org"],
            target_sr=self.config["sampling_rate"]["target"],
        )
        return sound

    def _extract_feature(self, sound):
        feature = self.feature_extractor(
            sound,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            sampling_rate=self.config["sampling_rate"]["target"],
            return_tensors="pt"
        )["input_values"].to(torch.float32)
        return feature

    def _read_labels(self, df):
        labels = torch.tensor(df.apply(self._to_onehot), dtype=torch.float32)
        return labels

    def _to_onehot(self, series):
        return [1 if l in series else 0 for l in self.config["labels"]]

################################################################################
# For Wav2Vec2 Classifier
################################################################################
class BirdClefBgClassifierDataset(BirdClefDataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.df = self._sample_df(df)
        self.filepaths = self._read_filepaths(self.df)
        self.feature_extractor = self._create_feature_extractor()
        self.labels = None
        if self.config["label"] in self.df.keys():
            self.labels = self._read_labels(self.df[self.config["label"]])
        self.transform = transform

    def _sample_df(self, df, false_ratio=1.0):
        df_false = df.loc[df[self.config["label"]].isin([["False"]])].reset_index(drop=True)
        num_false = len(df_false)
        num_true = np.round(num_false / false_ratio).astype(int)
        df_true = df.loc[df[self.config["label"]].isin([["True"]])].sample(num_true)
        df_sampled = pd.concat([df_false, df_true]).sample(frac=1, ignore_index=True)
        return df_sampled

################################################################################
# For Wav2Vec2 Prediction
################################################################################
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
        feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=self.config["sampling_rate"]["target"])
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
        wholesound, _ = librosa.load(
            filepath,
            sr=self.config["sampling_rate"]["org"]
        )
        if (self.config["sampling_rate"]["org"]==self.config["sampling_rate"]["target"]):
            return wholesound
        wholesound = librosa.resample(
            wholesound,
            orig_sr=self.config["sampling_rate"]["org"],
            target_sr=self.config["sampling_rate"]["target"]
        )
        return wholesound

    def _extract_feature(self, chunk):
        feature = self.feature_extractor(
            chunk,
            sampling_rate=self.config["sampling_rate"]["target"],
            return_tensors="pt"
        )["input_values"].to(torch.float32)
        return feature

################################################################################
# For EfficientNetBaseModel
################################################################################
class BirdClefMelspecDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.filepaths = self._read_filepaths(df)
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = self._read_labels(df[self.config["label"]])
        self.pre_transform = A.Compose(
            [A.Normalize(config["mean"], config["std"])]
        )
        self.to_tensor = Tv.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        melspec = self._read_melspec(self.filepaths[idx])
        melspec = self._normalize(melspec)
        melspec = self.pre_transform(image=melspec)["image"]
        melspec = self.to_tensor(melspec)
        if self.transform is not None:
            melspec = self.transform(melspec)
        if self.labels is not None:
            labels = self.labels[idx]
            return melspec, labels
        return melspec

    def _read_filepaths(self, df):
        values = df["filepath"].values
        return values

    def _read_melspec(self, filepath):
        melspec = np.load(filepath)["arr_0"]
        melspec = np.expand_dims(melspec, axis=-1)
        return melspec

    def _normalize(self, melspec, eps=1e-6):
        melspec = (melspec - melspec.mean()) / (melspec.std() + eps)
        if (melspec.max() - melspec.min()) < eps:
            return np.zeros_like(melspec, dtype=np.uint8)
        melspec = (
            255 * ((melspec - melspec.min()) / (melspec.max() - melspec.min()))
        ).astype(np.uint8)
        return melspec

    def _read_labels(self, df):
        labels = torch.tensor(df.apply(self._to_onehot), dtype=torch.float32)
        return labels

    def _to_onehot(self, series):
        return [1 if l in series else 0 for l in self.config["labels"]]

################################################################################
# For EfficientNetBaseModel(Context-rich Minority Oversampling)
################################################################################
class ContextRichMinorityOversampling:
    def __init__(self, df_train, config):
        self.config = config
        self.df_train = df_train
        self.labels, self.label_ratio = self._calc_label_ratio(df_train)

    def _calc_label_ratio(self, df_train):
        label_counts = df_train["labels"].value_counts()
        labels = label_counts.index
        inv_label_counts = [1/label_count for label_count in label_counts]
        label_ratio = inv_label_counts / np.sum(inv_label_counts)
        return labels, label_ratio

    def get(self):
        cutmix_ratio = np.random.beta(self.config["alpha"], self.config["alpha"])
        img_fg, label_fg = self._pickup_fg()
        img_bg, label_bg = self._pickup_bg()
        img, label = self._cutmix(img_fg, img_bg, label_fg, label_bg, cutmix_ratio)
        return img, label

    def _cutmix(self, img_fg, img_bg, label_fg, label_bg, ratio):
        cutwidth = int(img_fg.shape[1] * ratio)
        offset = np.random.randint(0, img_fg.shape[1] - cutwidth)
        img = img_bg
        img[:, offset:offset + cutwidth] = img_fg[:, offset:offset + cutwidth]
        label = label_fg * ratio + label_bg * (1 - ratio)
        label = torch.tensor(label, dtype=torch.float32)
        return img, label

    def _pickup_fg(self):
        labelnames_fg = np.random.choice(self.labels, p=self.label_ratio)
        filepath_fg = self.df_train.loc[
            self.df_train["labels"].str.get(0)==labelnames_fg[0], "filepath"
        ].sample(1).values[0]
        img_fg = np.load(filepath_fg)["arr_0"]
        label_fg = self._to_onehot(labelnames_fg)
        return img_fg, label_fg

    def _pickup_bg(self):
        sample_bg = self.df_train.sample(1)
        filepath_bg = sample_bg["filepath"].values[0]
        img_bg = np.load(filepath_bg)["arr_0"]
        labelnames_bg = sample_bg["labels"].values[0]
        label_bg = self._to_onehot(labelnames_bg)
        return img_bg, label_bg

    def _to_onehot(self, labels):
        return np.array([1 if l in labels else 0 for l in self.config["labels"]])

class BirdClefMelspecCMODataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.oversampling_rate = config["cmo"]["oversampling_rate"]
        self.cmo = ContextRichMinorityOversampling(df, self.config["cmo"])
        self.filepaths = self._read_filepaths(df)
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = self._read_labels(df[self.config["label"]])
        self.pre_transform = A.Compose(
            [A.Normalize(config["mean"], config["std"])]
        )
        self.to_tensor = Tv.ToTensor()
        self.transform = transform

    def __len__(self):
        if self.transform is None:
            return len(self.filepaths)
        if self.oversampling_rate <= 1.0:
            return len(self.filepaths)
        return int(len(self.filepaths)*self.oversampling_rate)

    def __getitem__(self, idx):
        if idx < len(self.filepaths):
            return self._pickup_org(idx)
        return self._get_oversample()

    def _pickup_org(self, idx):
        melspec = self._read_melspec(self.filepaths[idx])
        melspec = self._normalize(melspec)
        melspec = self.pre_transform(image=melspec)["image"]
        melspec = self.to_tensor(melspec)
        if self.transform is not None:
            melspec = self.transform(melspec)
        if self.labels is not None:
            labels = self.labels[idx]
            return melspec, labels
        return melspec

    def _get_oversample(self):
        melspec, labels = self.cmo.get()
        melspec = np.expand_dims(melspec, axis=-1)
        melspec = self._normalize(melspec)
        melspec = self.pre_transform(image=melspec)["image"]
        melspec = self.to_tensor(melspec)
        if self.transform is not None:
            melspec = self.transform(melspec)
        if self.labels is not None:
            return melspec, labels
        return melspec

    def _read_filepaths(self, df):
        values = df["filepath"].values
        return values

    def _read_melspec(self, filepath):
        melspec = np.load(filepath)["arr_0"]
        melspec = np.expand_dims(melspec, axis=-1)
        return melspec

    def _normalize(self, melspec, eps=1e-6):
        melspec = (melspec - melspec.mean()) / (melspec.std() + eps)
        if (melspec.max() - melspec.min()) < eps:
            return np.zeros_like(melspec, dtype=np.uint8)
        melspec = (
            255 * ((melspec - melspec.min()) / (melspec.max() - melspec.min()))
        ).astype(np.uint8)
        return melspec

    def _read_labels(self, df):
        labels = torch.tensor(df.apply(self._to_onehot), dtype=torch.float32)
        return labels

    def _to_onehot(self, series):
        return [1 if l in series else 0 for l in self.config["labels"]]

################################################################################
# For EfficientNetBase Prediction
################################################################################
class BirdClefMelspecPredDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.filepaths = self._read_filepaths(df)
        self.end_sec = self._read_end_sec(df)
        self.transform = transform

        # variables
        self.wholesound = None

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        chunk = self._read_chunk(self.filepaths[idx], self.end_sec[idx])
        melspec = self._to_melspec(chunk)
        if self.transform is not None:
            melspec = self.transform(melspec)
        return melspec

    def _read_filepaths(self, df):
        values = self.config["path"]["preddata"] \
            + df["row_id"].str.replace("_[0-9]+$", "", regex=True) \
            + ".ogg"
        return values

    def _read_end_sec(self, df):
        end_sec = df["row_id"].apply(lambda x: x.split("_")[-1]).astype(int)
        return end_sec

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
        wholesound, _ = librosa.load(
            filepath,
            sr=self.config["sampling_rate"]["org"]
        )
        if (self.config["sampling_rate"]["org"]==self.config["sampling_rate"]["target"]):
            return wholesound
        wholesound = librosa.resample(
            wholesound,
            orig_sr=self.config["sampling_rate"]["org"],
            target_sr=self.config["sampling_rate"]["target"]
        )
        return wholesound

    def _to_melspec(self, sound):
        melspec = librosa.feature.melspectrogram(
            y=sound, **self.config["melspec"]
        )
        return melspec

################################################################################
# For Pretraining(SimCLR)
################################################################################
class BirdClefMelspecSimCLRDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.filepaths = self._read_filepaths(df)
        self.pre_transform = A.Compose(
            [A.Normalize(config["mean"], config["std"])]
        )
        self.to_tensor = Tv.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        melspec = self._read_melspec(self.filepaths[idx])
        melspec = self._normalize(melspec)
        melspec = self.pre_transform(image=melspec)["image"]
        melspec = self.to_tensor(melspec)
        melspec_aug = self.transform(melspec)
        return melspec, melspec_aug

    def _read_filepaths(self, df):
        values = df["filepath"].values
        return values

    def _read_melspec(self, filepath):
        melspec = np.load(filepath)["arr_0"]
        melspec = np.expand_dims(melspec, axis=-1)
        return melspec

    def _normalize(self, melspec, eps=1e-6):
        melspec = (melspec - melspec.mean()) / (melspec.std() + eps)
        if (melspec.max() - melspec.min()) < eps:
            return np.zeros_like(melspec, dtype=np.uint8)
        melspec = (
            255 * ((melspec - melspec.min()) / (melspec.max() - melspec.min()))
        ).astype(np.uint8)
        return melspec

################################################################################
# For Pretraining(SimSiam)
################################################################################
class BirdClefMelspecSimSiamDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.filepaths = self._read_filepaths(df)
        self.pre_transform = A.Compose(
            [A.Normalize(config["mean"], config["std"])]
        )
        self.to_tensor = Tv.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        melspec = self._read_melspec(self.filepaths[idx])
        melspec = self._normalize(melspec)
        melspec = self.pre_transform(image=melspec)["image"]
        melspec = self.to_tensor(melspec)
        melspec_0 = self.transform(melspec)
        melspec_1 = self.transform(melspec)
        return melspec_0, melspec_1

    def _read_filepaths(self, df):
        values = df["filepath"].values
        return values

    def _read_melspec(self, filepath):
        melspec = np.load(filepath)["arr_0"]
        melspec = np.expand_dims(melspec, axis=-1)
        return melspec

    def _normalize(self, melspec, eps=1e-6):
        melspec = (melspec - melspec.mean()) / (melspec.std() + eps)
        if (melspec.max() - melspec.min()) < eps:
            return np.zeros_like(melspec, dtype=np.uint8)
        melspec = (
            255 * ((melspec - melspec.min()) / (melspec.max() - melspec.min()))
        ).astype(np.uint8)
        return melspec

################################################################################
# DataModule
################################################################################
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
        if (self.df_train is None) or (len(self.df_train) == 0):
            return None
        dataset = self.Dataset(
            self.df_train, self.config["dataset"], self.transforms["train"]
        )
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            **self.config["dataloader"]
        )
        return dataloader

    def val_dataloader(self):
        if (self.df_val is None) or (len(self.df_val) == 0):
            return None
        dataset = self.Dataset(
            self.df_val, self.config["dataset"], self.transforms["valid"]
        )
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            **self.config["dataloader"]
        )
        return dataloader

    def predict_dataloader(self):
        if (self.df_pred is None) or (len(self.df_pred) == 0):
            return None
        dataset = self.Dataset(
            self.df_pred, self.config["dataset"], self.transforms["pred"]
        )
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            **self.config["dataloader"]
        )
        return dataloader
