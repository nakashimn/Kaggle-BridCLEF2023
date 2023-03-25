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
from torch.nn import functional as F
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.sample import config
from components.preprocessor import DataPreprocessor
from components.datamodule import AudioClassificationDataset, BirdClefPredDataset

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
df_test = data_preprocessor.test_dataset()
df_pred = data_preprocessor.pred_dataset()

# FpDataSet
fix_seed(config["random_seed"])
dataset = AudioClassificationDataset(df_train, config["datamodule"]["dataset"])
batch = dataset.__getitem__(0)
