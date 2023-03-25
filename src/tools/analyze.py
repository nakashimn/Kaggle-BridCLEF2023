import os
import sys
import ast
import argparse
import pathlib
import glob
import datetime
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import traceback

# meta data
filepath_meta = "/workspace/kaggle/input/birdclef-2023/train_metadata.csv"
df_meta = pd.read_csv(filepath_meta)
value_counts = df_meta["primary_label"].value_counts()

df_meta["rating"].unique()

fig = plt.figure(figsize=[12, 2], tight_layout=True)
axis = fig.add_subplot(1, 1, 1)
axis.bar(value_counts.index, value_counts)

# check labels
primary_labels = sorted(df_meta["primary_label"].unique())
secondary_labels = sorted(
    np.unique(np.sum(
        [l for l in df_meta["secondary_labels"].apply(ast.literal_eval)]
    ))
)
df_primary_labels = pd.DataFrame(primary_labels, columns=["label"])
df_primary_labels["primary"] = True
df_secondary_labels = pd.DataFrame(secondary_labels, columns=["label"])
df_secondary_labels["secondary"] = True
df_labels = df_primary_labels.merge(df_secondary_labels, on="label", how="outer")
primary_only = df_labels.loc[df_labels["secondary"].isna(), "label"].unique()
secondary_only = df_labels.loc[df_labels["primary"].isna(), "label"].unique()

# check sounds
dirpath_audio = "/workspace/kaggle/input/birdclef-2023/train_audio/"
filepaths_audio = glob.glob(f"{dirpath_audio}/**/*.ogg", recursive=True)

snds, sampling_rate = librosa.load(filepaths_audio[0], sr=44100)

melspec = librosa.feature.melspectrogram(y=snds, sr=44100)
melspec_db = librosa.power_to_db(melspec)
