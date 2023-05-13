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

class StatsDration:
    def __init__(
        self,
        name,
        dirpath_input,
        dirpath_output,
        sampling_rate=32000,
        chunk_sec=5,
        ext="ogg"
    ):
        # const
        self.database_name = name
        self.sampling_rate = sampling_rate
        self.chunk_sec = chunk_sec
        self.ext = ext

        # path
        self.dirpath_audio = dirpath_input
        self.filepaths_audio = glob.glob(
            f"{self.dirpath_audio}/**/*.{self.ext}", recursive=True)
        self.dirpath_output = dirpath_output
        self.dirpath_output_table = f"{self.dirpath_output}/table/"
        self.dirpath_output_fig = f"{self.dirpath_output}/fig/"

        # variables
        df = None
        stats = None

    def __call__(self, write=True):
        self.run(write)

    def run(self, write=True):
        # table
        self.df = self._create_table()
        self.stats = self._calc_stats(self.df)
        if write:
            self.df.to_csv(
                f"{self.dirpath_output_table}/duration_{self.database_name}.csv", index=False)
            self.stats.to_csv(
                f"{self.dirpath_output_table}/stats_duration_{self.database_name}.csv", header=False)
        # fig
        fig = self._draw_fig(self.df)
        if write:
            fig.savefig(
                f"{self.dirpath_output_fig}/duration_{self.database_name}.png")

    def _create_table(self):
        durations = {
            "filename": [],
            "duration": []
        }
        for filepath_audio in tqdm(self.filepaths_audio):
            durations["filename"].append(
                pathlib.Path(filepath_audio).name
            )
            durations["duration"].append(
                librosa.get_duration(
                    path=filepath_audio,
                    sr=self.sampling_rate
                )
            )
        df_durations = pd.DataFrame(durations)
        return df_durations

    def _calc_stats(self, df):
        stats = self.df["duration"].describe()
        stats["1%"] = self.df["duration"].quantile(0.01)
        stats["99%"] = df["duration"].quantile(0.99)
        stats["< 5sec"] = (df["duration"]<5).sum()/len(df["duration"])
        stats["< 15sec"] = (df["duration"]<15).sum()/len(df["duration"])
        stats["< 30sec"] = (df["duration"]<30).sum()/len(df["duration"])
        stats["< 60sec"] = (df["duration"]<60).sum()/len(df["duration"])
        stats = stats.reindex(
            ["count", "mean", "std",
            "min", "1%", "25%", "50%", "75%", "99%", "max",
            "< 5sec", "< 15sec", "< 30sec", "< 60sec"]
        )
        return stats

    def _draw_fig(self, df):
        bins = np.arange(31)
        fig = plt.figure(figsize=[6, 2], tight_layout=True)
        axis = fig.add_subplot(1, 1, 1)
        axis = df["duration"].hist(ax=axis, bins=bins, grid=True)
        axis.set_xlabel("duration[sec]")
        axis.set_xlim([0, 30])
        axis.set_axisbelow(True)
        return fig


if __name__=="__main__":

    # EDA
    stats_duration_2022 = StatsDration(
        "birdclef_2022",
        dirpath_input="/workspace/kaggle/input/birdclef-2022/train_audio/",
        dirpath_output="/workspace/data/EDA/"
    )
    stats_duration_2022.run(write=True)

    stats_duration_2021 = StatsDration(
        "birdclef_2021",
        dirpath_input="/workspace/kaggle/input/birdclef-2021/train_short_audio/",
        dirpath_output="/workspace/data/EDA/"
    )
    stats_duration_2021.run(write=True)

    stats_duration_esc_50 = StatsDration(
        "ESC_50",
        dirpath_input="/workspace/data/ESC-50/audio/",
        dirpath_output="/workspace/data/EDA/",
        sampling_rate=44100,
        ext="wav"
    )
    stats_duration_esc_50.run(write=True)
