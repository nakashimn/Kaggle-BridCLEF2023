import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from transformers import ViTFeatureExtractor
import traceback

################################################################################
# Basic
################################################################################

class ImgRecogDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.config = config
        self.filepaths = self.read_filepaths(df)
        self.feature_extractor = self.create_feature_extractor()
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = self.read_labels(df)
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = read_image(self.filepaths[idx])
        if self.transform is not None:
            img = self.transform(img)
        feature = self.extract_feature(img)
        if self.labels is not None:
            labels = self.labels[idx]
            return feature, labels
        return feature

    def read_filepaths(self, df):
        values = df["filepath"].values
        return values

    def create_feature_extractor(self):
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            self.config["base_model_name"]
        )
        return feature_extractor

    def extract_feature(self, imgs):
        feature = self.feature_extractor(imgs, return_tensors="pt")
        feature["pixel_values"] = feature["pixel_values"].squeeze()
        return feature

    def read_labels(self, df):
        labels = F.one_hot(
            torch.tensor(
                [self.config["labels"].index(d) for d in df[self.config["label"]]]
            ),
            num_classes=self.config["num_class"]
        ).float()
        return labels

class DataModule(LightningDataModule):
    def __init__(
        self,
        df_train,
        df_val,
        df_pred,
        Dataset,
        config,
        transforms
    ):
        super().__init__()

        # const
        self.config = config
        self.df_train = df_train
        self.df_val = df_val
        self.df_pred = df_pred
        self.transforms = self.read_transforms(transforms)

        # class
        self.Dataset = Dataset

    def read_transforms(self, transforms):
        if transforms is not None:
            return transforms
        return {"train": None, "valid": None, "pred": None}

    def train_dataloader(self):
        dataset = self.Dataset(
            self.df_train,
            self.config["dataset"],
            self.transforms["train"]
        )
        return DataLoader(dataset, **self.config["train_loader"])

    def val_dataloader(self):
        dataset = self.Dataset(
            self.df_val,
            self.config["dataset"],
            self.transforms["valid"]
        )
        return DataLoader(dataset, **self.config["val_loader"])

    def predict_dataloader(self):
        dataset = self.Dataset(
            self.df_pred,
            self.config["dataset"],
            self.transforms["pred"]
        )
        return DataLoader(dataset, **self.config["pred_loader"])
