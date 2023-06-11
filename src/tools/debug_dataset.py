import os
import random
import sys
import ast
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import librosa
import transformers
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.trial_v5 import config
from components.preprocessor import DataPreprocessor
from components.datamodule import BirdClefDataset, BirdClefPredDataset, BirdClefMelspecDataset, DataModule

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

###
# sample
###
data_preprocessor = DataPreprocessor(config)
df_train = data_preprocessor.train_dataset_primary()
# df_train = data_preprocessor.train_dataset_for_bg_classifier()
df_test = data_preprocessor.test_dataset()
df_pred = data_preprocessor.pred_dataset_for_submit()

# FpDataSet
fix_seed(config["random_seed"])
# dataset = BirdClefPredDataset(df_pred, config["datamodule"]["dataset"])
# dataset = BirdClefDataset(df_train, config["datamodule"]["dataset"])
dataset = BirdClefMelspecDataset(df_train, config["datamodule"]["dataset"])
batch = dataset.__getitem__(0)

for i in tqdm(range(dataset.__len__())):
    batch = dataset.__getitem__(i)

dataloader = DataLoader(dataset, num_workers=0, batch_size=16, shuffle=False, drop_last=False)
for data in tqdm(dataloader):
    print(data)

datamodule = DataModule(
    df_train=df_train,
    df_val=None,
    df_pred=None,
    Dataset=BirdClefMelspecDataset,
    config=config,
    transforms=None
)
dir(datamodule)
datamodule.val_dataloader() is None

from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2FeatureExtractor
from torch import nn

duration_sec = 30
snd, _ = librosa.load(df_train["filepath"][0], duration=duration_sec)
spec = librosa.stft(
    y=snd,
    n_fft=5*32000,
    hop_length=32000
)
spec_db = librosa.power_to_db(spec)
librosa.display.specshow(spec_db)
feature_extractor = Wav2Vec2FeatureExtractor()
feature = feature_extractor(snd)
val = feature["input_values"]

val[0].shape
input_tensor = torch.tensor(val)

from config.trial_v3 import config
model_config = Wav2Vec2Config(**config["model"]["model_config"])
model = Wav2Vec2Model(model_config)


model.cpu()
model.eval()
result = model(input_tensor)
result.last_hidden_state[:, 1].shape
linear = nn.Linear(768, 265)
res = linear(result.last_hidden_state[:, 0])
res.shape

df = pd.read_csv("/workspace/kaggle/input/birdclef-2023-oversampled-5sec/train_metadata_v1_1.csv")
df_ = pd.read_csv("/workspace/kaggle/input/birdclef-2023-oversampled-5sec/train_metadata_v1.csv")
dirpath_melspec = "/workspace/kaggle/input/birdclef-2023-oversampled-5sec/train_melspec_v1/"
df["filepath"] = dirpath_melspec + df["filename"]

for idx in tqdm(df.index):
    if not os.path.exists(df.loc[idx, "filepath"]):
        print(df.loc[idx, "filepath"])

df_append = df_.loc[df_["end_sec"].isna()].reset_index(drop=True)

df_new = pd.concat([df, df_append]).drop_duplicates(ignore_index=True)

len(df_new)

df_new.to_csv("/workspace/kaggle/input/birdclef-2023-oversampled-5sec/train_metadata_v1_2.csv", index=False)
df_new["primary_label"].value_counts()

df_comp = pd.merge(df_new, df_, on=list(df_new.columns), how="outer", indicator=True)

df_new["filename"].str.split("/", expand=True)[0].unique()


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
        cutmix_ratio = np.random.uniform(0.5, 1.0)
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

crm = ContextRichMinorityOversampling(df_train, config)
img, label = crm.get()
