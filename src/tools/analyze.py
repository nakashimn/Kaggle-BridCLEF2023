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
from IPython.display import Audio
import traceback

# meta data
dirpath_dataset = "/workspace/kaggle/input/birdclef-2023/"
dirpath_dataset_modified = "/workspace/kaggle/input/birdclef-2023-modified/"
filepath_meta = f"{dirpath_dataset}/train_metadata.csv"
df_meta = pd.read_csv(filepath_meta)


# check labels
value_counts = df_meta["primary_label"].value_counts()
fig = plt.figure(figsize=[12, 2], tight_layout=True)
axis = fig.add_subplot(1, 1, 1)
axis.bar(value_counts.index, value_counts)

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

# check duration
# df_meta["duration"] = None
# for idx in tqdm(df_meta["filename"].index):
#     filepath = f"{dirpath_audio}/{df_meta.loc[idx, 'filename']}"
#     df_meta.loc[idx, "duration"] = librosa.get_duration(filename=filepath, sr=32000)

# df_meta.to_csv(f"{dirpath_dataset_modified}/train_metadata.csv", index=False)

df_meta_modified = pd.read_csv(f"{dirpath_dataset_modified}/train_metadata.csv")
df_meta_modified = df_meta_modified.sort_values("duration", ignore_index=True, ascending=False)

threshold_duration = df_meta_modified["duration"].quantile(0.99)
df_meta_long = df_meta_modified.loc[df_meta_modified["duration"]>threshold_duration]

df_meta_long.iloc[60:]["duration"].sum()/6
# Audio(f"{dirpath_audio}/{df_meta_long.iloc[-1]['filename']}")

sampling_rate = 32000
margin_sec = 5
duration_sec = 30
threshold_ratio = 0.8
lowpass_window_sec = 1
lowpass_filter = np.ones(lowpass_window_sec * sampling_rate)/(lowpass_window_sec * sampling_rate)
for idx in df_meta_long.index[60:]:
    filepath = f"{dirpath_audio}/{df_meta_long.loc[idx, 'filename']}"
    sound, _ = librosa.load(filepath, sr=sampling_rate)
    lowpass_abs_sound = np.convolve(np.abs(sound), lowpass_filter)
    threshold_sound = np.quantile(lowpass_abs_sound, threshold_ratio)

    num_chunk = np.ceil(len(lowpass_abs_sound)/(duration_sec*sampling_rate)).astype(int)
    sounds = [sound[i*duration_sec*sampling_rate:(i+1)*duration_sec*sampling_rate] for i in range(num_chunk)]
    lowpass_abs_sounds = [lowpass_abs_sound[i*duration_sec*sampling_rate:(i+1)*duration_sec*sampling_rate] for i in range(num_chunk)]
    idx_active_chunk = [np.max(s) >= threshold_sound for s in lowpass_abs_sounds]
    active_sounds = [sounds[i] for i in range(num_chunk) if idx_active_chunk[i]]
    inactive_sounds = [sounds[i] for i in range(num_chunk) if ~idx_active_chunk[i]]

    Audio(sound, rate=sampling_rate)

    Audio(sound[::6], rate=sampling_rate)
