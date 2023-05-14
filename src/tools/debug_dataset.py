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
from config.sample import config
# from config.bg_classify_v0 import config
from components.preprocessor import DataPreprocessor
from components.datamodule import BirdClefDataset, BirdClefPredDataset, BirdClefMelspecDataset

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
