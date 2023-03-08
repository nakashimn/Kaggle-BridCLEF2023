import os
import sys
import shutil
from tqdm import tqdm
import pathlib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.sample import config
from components.preprocessor import TextCleaner, DataPreprocessor

###
# sample
###

# prepare
tokenizer = AutoTokenizer.from_pretrained(
    config["datamodule"]["dataset"]["base_model_name"],
    use_fast=config["datamodule"]["dataset"]["use_fast_tokenizer"]
)
df_train = pd.read_csv(config["path"]["traindata"])

data_preprocessor = DataPreprocessor(config)
df_train = data_preprocessor.train_dataset()

# tokenize
for idx in tqdm(df_train.index):
    text = df_train.loc[idx, "discourse_type"]
    try:
        token = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=config["datamodule"]["dataset"]["max_length"],
            padding="max_length"
        )
        ids = torch.tensor([token["input_ids"]])
        masks = torch.tensor([token["attention_mask"]])
    except:
        print(text)
        print(traceback.format_exc())
