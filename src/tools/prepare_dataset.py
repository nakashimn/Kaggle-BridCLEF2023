import os
import sys
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

if __name__=="__main__":
    # param
    sampling_rate = 32000
    chunk_sec = 5

    # path
    dirpath_dataset_org = "/workspace/kaggle/input/birdclef-2023/"
    dirpath_dataset_modified = "/kaggle/input/birdclef-2023-modified-5sec/"
    dirname = "/train_audio/"
    filepaths_audio = glob.glob(f"{dirpath_dataset_org}/{dirname}/*/*.ogg")

    # split soundfile
    for filepath in tqdm(filepaths_audio):
        label = pathlib.Path(filepath).parent.name
        os.makedirs(f"{dirpath_dataset_modified}/{dirname}/{label}/", exist_ok=True)
        filestem = pathlib.Path(filepath).stem

        snd, _ = librosa.load(filepath, sr=sampling_rate)
        snd_sec = librosa.get_duration(filename=filepath, sr=sampling_rate)
        num_chunk = int(np.ceil(snd_sec / chunk_sec))

        for i in tqdm(range(1, num_chunk+1), leave=False):
            filename = f"{filestem}_{int(chunk_sec*i)}.npz"
            chunk = snd[
                sampling_rate * int(chunk_sec*(i-1)):sampling_rate * int(chunk_sec*i)
            ]
            np.savez_compressed(f"{dirpath_dataset_modified}/{dirname}/{label}/{filename}", chunk)

    # create meta table
    filepath_meta_org = f"{dirpath_dataset_modified}/train_metadata_org.csv"
    df_meta = pd.read_csv(filepath_meta_org)
    df_meta = df_meta.rename(columns={"filename": "filename_org"})
    filepaths_audio = glob.glob(f"{dirpath_dataset_modified}/{dirname}/**/*.npz")
    split_data_meta = {
        "filename": [f"{pathlib.Path(f).parent.name}/{pathlib.Path(f).name}" for f in filepaths_audio],
        "filename_org": [f"{pathlib.Path(f).parent.name}/{pathlib.Path(f).name.split('_')[0]}.ogg" for f in filepaths_audio]
    }
    df_split_meta = pd.DataFrame(split_data_meta).merge(df_meta, on="filename_org")
    df_split_meta.to_csv(f"{dirpath_dataset_modified}/train_metadata.csv", index=False)
