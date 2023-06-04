import os
import sys
import shutil
import argparse
import pathlib
import glob
import datetime
import json
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import traceback

class SplitSoundFile:
    dirpath_src = ""
    dirpath_dst = ""
    filepaths = []
    sampling_rate = 32000
    chunk_sec = 5

    def __init__(self, dirpath_src, dirpath_dst, ext_src="ogg", num_process=8):
        self.__class__.dirpath_src = dirpath_src
        self.__class__.dirpath_dst = dirpath_dst
        self.pool = Pool(num_process)
        self.__class__.filepaths = glob.glob(
            f"{dirpath_src}/**/*.{ext_src}", recursive=True
        )

    def run(self):
        pbar = tqdm(total=len(self.filepaths))
        for _ in self.pool.imap_unordered(self._split, self.filepaths):
            pbar.update(1)

    @classmethod
    def _split(cls, filepath):
        label = pathlib.Path(filepath).parent.name
        os.makedirs(f"{cls.dirpath_dst}/{label}/", exist_ok=True)
        filestem = pathlib.Path(filepath).stem

        snd, _ = librosa.load(filepath, sr=cls.sampling_rate)
        snd_sec = librosa.get_duration(filename=filepath, sr=cls.sampling_rate)
        num_chunk = int(np.ceil(snd_sec / cls.chunk_sec))

        for i in tqdm(range(1, num_chunk+1), leave=False):
            filename = f"{filestem}_{int(cls.chunk_sec*i)}.npz"
            chunk = snd[
                cls.sampling_rate * int(cls.chunk_sec*(i-1)):cls.sampling_rate * int(cls.chunk_sec*i)
            ]
            np.savez_compressed(f"{cls.dirpath_dst}/{label}/{filename}", chunk)
        return True


class ConvertSoundToMelspec:
    dirpath_src = ""
    dirapth_dst = ""
    filepaths = []

    sampling_rate = 32000
    chunk_sec = 5
    n_mels = 256
    fmin = 16
    fmax = 16386
    n_fft = 2048
    hop_length = 512

    def __init__(self, dirpath_src, dirpath_dst, num_process=8):
        self.__class__.dirpath_src = dirpath_src
        self.__class__.dirpath_dst = dirpath_dst
        self.pool = Pool(num_process)
        self.__class__.filepaths = glob.glob(
            f"{dirpath_src}/**/*.npz", recursive=True
        )

    def run(self):
        pbar = tqdm(total=len(self.filepaths))
        for _ in self.pool.imap_unordered(self._convert_snd_to_melspec, self.filepaths):
            pbar.update(1)

    @classmethod
    def _convert_snd_to_melspec(cls, filepath):
        label = pathlib.Path(filepath).parent.name
        os.makedirs(f"{cls.dirpath_dst}/{label}/", exist_ok=True)
        filename = pathlib.Path(filepath).name
        filepath_output = f"{cls.dirpath_dst}/{label}/{filename}"

        # check existing files
        if os.path.exists(filepath_output): return False

        # load
        snd = np.load(f"{filepath}")["arr_0"]

        # compensate
        if len(snd) < (cls.chunk_sec * cls.sampling_rate): return False

        # transform
        melspec = librosa.feature.melspectrogram(
            y=snd,
            sr=cls.sampling_rate,
            n_mels=cls.n_mels,
            n_fft=cls.n_fft,
            hop_length=cls.hop_length,
            fmin=cls.fmin,
            fmax=cls.fmax
        ).astype(np.float32)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)

        # output
        np.savez_compressed(filepath_output, melspec_db)
        return True


if __name__=="__main__":
    # path
    dirpath_dataset_org = "/workspace/kaggle/input/birdclef-2023/"
    dirpath_audio_org = f"{dirpath_dataset_org}/train_audio/"
    filepath_meta_org = f"{dirpath_dataset_org}/train_metadata.csv"
    dirpath_dataset_output = "/kaggle/input/birdclef-2023-modified/"
    dirpath_audio_output = f"{dirpath_dataset_output}/train_audio/"
    dirpath_melspec_output = f"{dirpath_dataset_output}/train_melspec/"

    os.makedirs(f"{dirpath_audio_output}", exist_ok=True)
    os.makedirs(f"{dirpath_melspec_output}", exist_ok=True)

    # split soundfile
    split_snd = SplitSoundFile(
        dirpath_src=dirpath_audio_org,
        dirpath_dst=dirpath_audio_output
    )
    split_snd.run()

    # convert sound to melspec
    convert_snd_to_melspec = ConvertSoundToMelspec(
        dirpath_src=dirpath_audio_output,
        dirpath_dst=dirpath_melspec_output
    )
    convert_snd_to_melspec.run()

    # create meta table
    df_meta = pd.read_csv(filepath_meta_org)
    df_meta = df_meta.rename(columns={"filename": "filename_org"})
    filepaths_audio = glob.glob(f"{dirpath_audio_output}/**/*.npz", recursive=True)
    split_data_meta = {
        "filename": [f"{pathlib.Path(f).parent.name}/{pathlib.Path(f).name}" for f in filepaths_audio],
        "filename_org": [f"{pathlib.Path(f).parent.name}/{pathlib.Path(f).name.split('_')[0]}.ogg" for f in filepaths_audio]
    }
    df_split_meta = pd.DataFrame(split_data_meta).merge(df_meta, on="filename_org")
    df_split_meta.to_csv(f"{dirpath_dataset_output}/train_metadata.csv", index=False)
