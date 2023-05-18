import os
import sys
import pathlib
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from pytorch_lightning import LightningModule
import timm
from transformers import Wav2Vec2Model, Wav2Vec2ForPreTraining, Wav2Vec2Config, PretrainedConfig, PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from lightly.models.modules import SimCLRProjectionHead, SimSiamProjectionHead, SimSiamPredictionHead
from lightly.loss import NTXentLoss, NegativeCosineSimilarity
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from loss_functions import FocalLoss
from augmentation import Mixup, LabelSmoothing

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
        if self.config["base_model_name"] is not None:
            model = Wav2Vec2Model.from_pretrained(self.config["base_model_name"])
        else:
            model = Wav2Vec2Model(
                Wav2Vec2Config(**self.config["model_config"])
            )
        classifier = nn.Sequential(
            nn.Linear(
                model.config.hidden_size,
                self.config["fc_feature_dim"],
                bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                self.config["fc_feature_dim"],
                self.config["num_class"],
                bias=True
            )
        )
        if self.config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        if self.config["freeze_base_model"]:
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

class BirdClefPretrainModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.model = self._create_model()

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    def _create_model(self):
        if self.config["base_model_name"] is not None:
            model = Wav2Vec2Model.from_pretrained(self.config["base_model_name"])
        else:
            model = Wav2Vec2ForPreTraining(
                Wav2Vec2Config(**self.config["model_config"])
            )
        model = model.train()
        if self.config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        return model

    def forward(self, features, mask_time_indices, sampled_negative_indices):
        out = self.model(features, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices)
        return out

    def training_step(self, batch, batch_idx):
        features = batch
        input_values = features.squeeze(dim=1).to(self.model.device)
        batch_size, raw_sequence_length = input_values.shape
        sequence_length = self.model._get_feat_extract_output_lengths(raw_sequence_length).item()
        mask_time_indices = _compute_mask_indices(
            shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
        )
        sampled_negative_indices = _sample_negative_indices(
            features_shape=(batch_size, sequence_length),
            num_negatives=self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
        sampled_negative_indices = torch.tensor(
            data=sampled_negative_indices, device=input_values.device, dtype=torch.long
        )
        out = self.forward(input_values, mask_time_indices, sampled_negative_indices)
        return {"loss": out.loss}

    def training_epoch_end(self, outputs):
        metrics = torch.tensor([out["loss"] for out in outputs]).mean()
        self.min_loss = np.nanmin([self.min_loss, metrics.detach().cpu().numpy()])
        self.log(f"train_loss", metrics)

        return super().training_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

    def save_pretrained_model(self):
        self.model.save_pretrained(save_directory=self.config["save_directory"])

class BirdClefBgClassifierModel(LightningModule):
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
        if self.config["base_model_name"] is not None:
            model = Wav2Vec2Model.from_pretrained(self.config["base_model_name"])
        else:
            model = Wav2Vec2Model(
                Wav2Vec2Config(**self.config["model_config"])
            )
        classifier = nn.Sequential(
            nn.Linear(
                model.config.hidden_size,
                self.config["fc_feature_dim"],
                bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                self.config["fc_feature_dim"],
                self.config["num_class"],
                bias=True
            )
        )
        if self.config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        if self.config["freeze_base_model"]:
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

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

class AttBlockV2(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation="linear"
    ):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class TimmSEDBaseConfig(PretrainedConfig):
    effnet_pretrained = True

class TimmSEDBaseModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.effnet_pretrained = config.effnet_pretrained
        self.bn0, self.encoder = self._create_model()
        self.num_features = self.encoder[-1].num_features

    def _create_model(self):
        # batch normalization
        bn0 = nn.BatchNorm2d(256)
        # encoder
        base_model = timm.create_model(
            "tf_efficientnet_b4_ns",
            pretrained=self.effnet_pretrained,
            num_classes=0,
            global_pool="",
            in_chans=1
        )
        layers = list(base_model.children())[:-2]
        encoder = nn.Sequential(*layers)
        return bn0, encoder

    def forward(self, input_data):

        x = input_data[:, [0], :, :]

        # normalize over mel_freq
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        x = self.encoder(x)

        x = torch.mean(x, dim=2)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        return x

class BirdClefTimmSEDModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.base_model, self.linear, self.att_block = self._create_model()
        self.criterion = eval(config["loss"]["name"])(**self.config["loss"]["params"])

        # augmentation
        self.mixup = Mixup(config["mixup"]["alpha"])
        self.label_smoothing = LabelSmoothing(
            config["label_smoothing"]["eps"], config["num_class"]
        )

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    def _create_model(self):
        # basemodel
        if self.config["base_model_name"] is not None:
            base_model = TimmSEDBaseModel.from_pretrained(
                self.config["base_model_name"],
                config=TimmSEDBaseConfig()
            )
        else:
            base_model = TimmSEDBaseModel(TimmSEDBaseConfig())

        # linear
        linear = nn.Linear(
            base_model.num_features,
            base_model.num_features,
            bias=True
        )
        # attention block
        att_block = AttBlockV2(
            base_model.num_features,
            self.config["num_class"],
            activation="sigmoid"
        )
        return base_model, linear, att_block

    def forward(self, input_data):
        x = self.base_model(input_data)

        x = x.transpose(1, 2)
        x = F.relu(self.linear(x))
        x = x.transpose(1, 2)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        return clipwise_output

    def training_step(self, batch, batch_idx):
        melspec, labels = batch
        melspec, labels = self.mixup(melspec, labels)
        labels = self.label_smoothing(labels)
        logits = self.forward(melspec)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "label": label}

    def validation_step(self, batch, batch_idx):
        melspec, labels = batch
        logits = self.forward(melspec)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        prob = logits.softmax(axis=1).detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "prob": prob, "label": label}

    def predict_step(self, batch, batch_idx):
        melspec = batch
        logits = self.forward(melspec)
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

class BirdClefTimmSEDSimCLRModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.base_model, self.projection = self._create_model()
        self.criterion = NTXentLoss()

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    def _create_model(self):
        # encoder
        base_model = TimmSEDBaseModel()

        projection = SimCLRProjectionHead(base_model.num_features*10, 2048, 2048)
        # # linear
        # linear = nn.Linear(
        #     base_model.num_features,
        #     base_model.num_features,
        #     bias=True
        # )
        # # attention block
        # att_block = AttBlockV2(
        #     base_model.num_features,
        #     self.config["num_class"],
        #     activation="sigmoid"
        # )
        return base_model, projection

    def forward(self, input_data):
        x = self.base_model(input_data)
        melspec = self.projection(x.flatten(start_dim=1))
        return melspec

    def training_step(self, batch, batch_idx):
        melspec_0, melspec_1 = batch
        logits_0 = self.forward(melspec_0)
        logits_1 = self.forward(melspec_1)
        loss = self.criterion(logits_0, logits_1)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        metrics = torch.tensor([out["loss"] for out in outputs]).mean()
        self.log(f"train_loss", metrics)

        return super().training_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

    def save_pretrained_model(self):
        self.base_model.save_pretrained(save_directory=self.config["save_directory"])

class BirdClefTimmSEDSimSiamModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.base_model, self.projection, self.prediction = self._create_model()
        self.criterion = NegativeCosineSimilarity()

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    def _create_model(self):
        # encoder
        dummy_config = TimmSEDBaseConfig()
        base_model = TimmSEDBaseModel(dummy_config)

        projection = SimSiamProjectionHead(base_model.num_features*10, 2048, 2048)
        prediction = SimSiamPredictionHead(2048 ,2048, 2048)
        return base_model, projection, prediction

    def forward(self, input_data):
        x = self.base_model(input_data)
        z = self.projection(x.flatten(start_dim=1))
        p = self.prediction(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        melspec_0, melspec_1 = batch
        z_0, p_0 = self.forward(melspec_0)
        z_1, p_1 = self.forward(melspec_1)
        loss = 0.5 * (self.criterion(z_0, p_1) + self.criterion(z_1, p_0))
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        metrics = torch.tensor([out["loss"] for out in outputs]).mean()
        self.log(f"train_loss", metrics)

        return super().training_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

    def save_pretrained_model(self):
        self.base_model.save_pretrained(save_directory=self.config["save_directory"])
