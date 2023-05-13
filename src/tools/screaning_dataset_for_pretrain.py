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
import matplotlib.pyplot as plt
import traceback

if __name__=="__main__":

    # path
    dirpath_dataset_2021 = "/kaggle/input/birdclef-2021-modified/train_audio/"
    dirpath_dataset_2022 = "/kaggle/input/birdclef-2022-modified/train_audio/"
    dirpath_dataset_esc50 = "/kaggle/input/ESC-50-modified/train_audio/"
    filepaths_dataset_2021 = glob.glob(f"{dirpath_dataset_2021}/*.npz")
    filepaths_dataset_2022 = glob.glob(f"{dirpath_dataset_2022}/*.npz")
    filepaths_dataset_esc50 = glob.glob(f"{dirpath_dataset_esc50}/*.npz")
    dirpath_output = "/kaggle/input/pretrain_dataset_5s_v0/train_audio/"

    # check duplication
    fileids_2021 = np.unique([pathlib.Path(f).stem.split("_")[0] for f in filepaths_dataset_2021])
    fileids_2022 = np.unique([pathlib.Path(f).stem.split("_")[0] for f in filepaths_dataset_2022])

    df_fileids_2021 = pd.DataFrame({"file_id": fileids_2021, "2021": True})
    df_fileids_2022 = pd.DataFrame({"file_id": fileids_2022, "2022": True})

    df_fileids = df_fileids_2021.merge(df_fileids_2022, on="file_id", how="outer").fillna(False)
    duplicated_ids = df_fileids.loc[df_fileids["2021"] & df_fileids["2022"], "file_id"].values

    #
    for filepath in tqdm(filepaths_dataset_esc50):
        filename = pathlib.Path(filepath).name
        os.symlink(filepath, f"{dirpath_output}{filename}")

    for filepath in tqdm(filepaths_dataset_2022):
        filename = pathlib.Path(filepath).name
        os.symlink(filepath, f"{dirpath_output}{filename}")

    filenames_duplicated = []
    for filepath in tqdm(filepaths_dataset_2021):
        filename = pathlib.Path(filepath).name
        if os.path.exists(f"{dirpath_output}{filename}"):
            filenames_duplicated.append(filename)
            continue
        os.symlink(filepath, f"{dirpath_output}{filename}")

    # df_duplicated_filenames = pd.DataFrame({"filename": filenames_duplicated})
    # df_duplicated_filenames.to_csv("/workspace/data/EDA/table/duplicated_filename.csv", index=False)
