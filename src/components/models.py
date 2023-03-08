import os
import sys
import pathlib
from abc import ABCMeta, abstractmethod
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from pytorch_lightning import LightningModule
from transformers import ViTForImageClassification
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from loss_functions import FocalLoss

################################################################################
# Base Class
################################################################################

class ImgRecogModelBase(LightningModule, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.base_model = self.create_base_model()
        self.dropout = nn.Dropout(self.config["dropout_rate"])

        self.criterion = eval(config["loss"]["name"])(
            **self.config["loss"]["params"]
        )

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    @abstractmethod
    def create_base_model(self):
        pass

    @abstractmethod
    def forward(self, imgs):
        pass

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self.forward(features)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "label": label}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self.forward(features)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        prob = logits.softmax(axis=1).detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "prob": prob, "label": label}

    def predict_step(self, batch, batch_idx):
        features = batch
        logits = self.forward(features)
        prob = logits.softmax(axis=1).detach()
        return {"prob": prob}

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(),
            **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer,
            **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

################################################################################
# Basic
################################################################################

class ImgRecogModel(ImgRecogModelBase):
    def __init__(self, config):
        super().__init__(config)

        # const
        self.fc = self.create_fully_connected()

    def create_base_model(self):
        base_model = ViTForImageClassification.from_pretrained(
            self.config["base_model_name"]
        )
        if not self.config["freeze_base_model"]:
            return base_model
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model

    def create_fully_connected(self):
        return nn.Linear(self.config["dim_feature"], self.config["num_class"])

    def forward(self, features):
        out = self.base_model(**features)
        out = self.dropout(out[0])
        out = self.fc(out)
        return out

    def training_epoch_end(self, outputs):
        logits = torch.cat([out["logit"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        metrics = self.criterion(logits, labels)
        self.min_loss = np.nanmin([self.min_loss, metrics.detach().cpu().numpy()])
        self.log(f"train_loss", metrics)

        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        logits = torch.cat([out["logit"] for out in outputs])
        probs = torch.cat([out["prob"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        metrics = self.criterion(logits, labels)
        self.log(f"val_loss", metrics)

        self.val_probs = probs.detach().cpu().numpy()
        self.val_labels = labels.detach().cpu().numpy()

        return super().validation_epoch_end(outputs)
