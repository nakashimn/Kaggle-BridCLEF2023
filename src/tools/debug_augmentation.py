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
from IPython.display import Audio

if __name__=="__main__":

    # const
    sampling_rate = 32000

    # path
    dirpath_audio = "/workspace/kaggle/input/birdclef-2022-modified/train_audio/"
    filepaths_audio = glob.glob(f"{dirpath_audio}/*.npz")


    snd = np.load(filepaths_audio[0])["arr_0"]

    # base
    Audio(snd, rate=32000)

    # [x] pitch shift [-0.5, 0.5]
    snd_pitch_shifted = librosa.effects.pitch_shift(snd, n_steps=0, sr=32000)
    Audio(snd_pitch_shifted, rate=32000)

    # [ ] time stretch
    snd_time_stretch = librosa.effects.time_stretch(snd, rate=0.9)
    Audio(snd_time_stretch, rate=32000)

    # [x] hermonic [1, 3]
    snd_harmonic = librosa.effects.harmonic(snd, margin=3)
    Audio(snd_harmonic, rate=32000)

    # [x] percussive [1, 3]
    snd_percussive = librosa.effects.percussive(snd, margin=1)
    Audio(snd_percussive, rate=32000)
