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
from augmentations import Mixup, LabelSmoothing
from validations import CMeanAveragePrecision

################################################################################
# Wav2Vec2Base Model
################################################################################
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

################################################################################
# Wav2Vec2Base Pretrain Model
################################################################################
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

################################################################################
# Wav2Vec2Base Classifier Model
################################################################################
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

################################################################################
# EfficientNet
################################################################################
class BirdClefEfficientNetModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.bn, self.encoder, self.fc = self._create_model()
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
        # batch_normalization
        bn = nn.BatchNorm2d(self.config["n_mels"])
        # basemodel
        base_model = timm.create_model(
            self.config["base_model_name"],
            pretrained=True,
            num_classes=0,
            global_pool="",
            in_chans=1
        )
        layers = list(base_model.children())[:-2]
        encoder = nn.Sequential(*layers)
        # linear
        fc = nn.Sequential(
            nn.Linear(
                encoder[-1].num_features * 10,
                self.config["fc_mid_dim"],
                bias=True
            ),
            nn.ReLU(),
            nn.Linear(
                self.config["fc_mid_dim"],
                self.config["num_class"],
                bias=True
            )
        )
        return bn, encoder, fc

    def forward(self, input_data):
        x = input_data[:, [0], :, :]
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.mean(dim=2)
        x = x.flatten(start_dim=1)
        out = self.fc(x)
        return out

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

        cmap = CMeanAveragePrecision(self.val_probs, self.val_labels, {"padding_num": 5}).calc()
        self.log("cmAP", cmap)

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

################################################################################
# Attention Block
################################################################################
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

################################################################################
# SED Model
################################################################################
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
            "tf_efficientnet_b2_ns",
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
            activation="linear"
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

        cmap = CMeanAveragePrecision(self.val_probs, self.val_labels, {"padding_num": 5}).calc()
        self.log("cmAP", cmap)

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

################################################################################
# SED Model for Pretraining(SimCLR)
################################################################################

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

################################################################################
# SED Model for Pretraining(SimSiam)
################################################################################
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
        self.collapse_level = np.nan

    def _create_model(self):
        # encoder
        dummy_config = TimmSEDBaseConfig()
        base_model = TimmSEDBaseModel(dummy_config)

        projection = SimSiamProjectionHead(
            base_model.num_features*10,
            self.config["projection_hidden_dim"],
            self.config["out_dim"]
        )
        prediction = SimSiamPredictionHead(
            self.config["out_dim"],
            self.config["projection_hidden_dim"],
            self.config["out_dim"]
        )
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

        #
        output_norm = torch.nn.functional.normalize(p_0.detach(), dim=1)
        return {"loss": loss, "output_norm": output_norm}

    def training_epoch_end(self, outputs):
        train_loss = torch.tensor([out["loss"] for out in outputs]).mean()
        self.log(f"train_loss", train_loss)

        output_std = torch.concat([out["output_norm"] for out in outputs], dim=0).std(dim=0)
        output_std_mean = output_std.mean().to("cpu").numpy()
        collapse_level = np.max([0.0, 1.0 - np.sqrt(self.config["out_dim"]) * output_std_mean])
        self.log(f"collapse_level", collapse_level)
        self.collapse_level = collapse_level

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
