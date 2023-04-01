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
from components.preprocessor import DataPreprocessor
from components.datamodule import BirdClefDataset, BirdClefPredDataset

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
df_pred = data_preprocessor.pred_dataset_for_submit()

# FpDataSet
fix_seed(config["random_seed"])
dataset = BirdClefPredDataset(df_pred, config["datamodule"]["dataset"])
batch = dataset.__getitem__(119)
for i in tqdm(range(dataset.__len__())):
    batch = dataset.__getitem__(i)

dataloader = DataLoader(dataset, num_workers=0, batch_size=4, shuffle=False, drop_last=False)
for data in tqdm(dataloader):
    print(data)
