import os
import sys
import pathlib
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from pytorch_lightning import LightningModule
from transformers import Wav2Vec2Model, Wav2Vec2Config
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from loss_functions import FocalLoss

class BirdClefModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.model, self.classifier = self._create_model()
        self.criterion = eval(config["loss"]["name"])(**self.config["loss"]["params"])

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    def _create_model(self):
        config = Wav2Vec2Config(**self.config["model_config"])
        model = Wav2Vec2Model(config)
        classifier = nn.Linear(
            model.config.hidden_size,
            self.config["num_class"],
            bias=True
        )
        if not self.config["freeze_base_model"]:
            return model, classifier
        for param in model.encoder.parameters():
            param.requires_grad = False
        return model, classifier

    def forward(self, features):
        out = self.model(features.squeeze(dim=1).to(self.model.device))
        logits = self.classifier(out.last_hidden_state[:, 0])
        return logits

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

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]
