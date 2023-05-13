import os
import pathlib
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import traceback

class SplitAudio:

  def __init__(
      self,
      dirpath_input,
      dirpath_output,
      sampling_rate_org=32000,
      sampling_rate_target=32000,
      chunk_sec=5,
      ext="ogg",
      random_shift=True
    ):
    # const
    self.sampling_rate_org = sampling_rate_org
    self.sampling_rate_target = sampling_rate_target
    self.chunk_sec = chunk_sec
    self.ext = ext
    self.random_shift = random_shift

    # path
    self.dirpath_input = dirpath_input
    self.dirpath_output = dirpath_output
    self.filepaths = glob.glob(f"{dirpath_input}/**/*.{ext}", recursive=True)

    # variables
    self.df = None

  def run(self):
    data_info = {
      "filename": []
    }
    for filepath in tqdm(self.filepaths):
      filestem = pathlib.Path(filepath).stem
      snd, _ = librosa.load(path=filepath, sr=self.sampling_rate_org)
      if self.sampling_rate_org!=self.sampling_rate_target:
        snd = librosa.resample(
            snd,
            orig_sr=self.sampling_rate_org,
            target_sr=self.sampling_rate_target
        )
      chunks = self._split_audio(snd)
      for i, chunk in tqdm(enumerate(chunks), leave=False):
        filename_output = f"{filestem}_{(i+1)*self.chunk_sec}.npz"
        data_info["filename"].append(filename_output)
        np.savez_compressed(
          f"{self.dirpath_output}/{filename_output}", chunk
        )
    self.df = pd.DataFrame(data_info)
    return self.df

  def _split_audio(self, snd):
    duration_sec = len(snd) / self.sampling_rate_target
    # use whole audio
    if duration_sec <= 5.0:
      chunks = [snd]
      return chunks
    # split audio into chunks
    if self.random_shift:
      snd = self.shift_offset(snd, duration_sec)
    num_chunk = int(duration_sec // self.chunk_sec)
    chunks = np.split(
      snd,
      [int(self.chunk_sec*self.sampling_rate_target*i) for i in range(1, num_chunk+1)]
    )[:num_chunk]
    return chunks

  def shift_offset(self, snd, duration_sec):
    residue_sec = duration_sec % self.chunk_sec
    offset_sec = np.random.uniform(0, 1) * residue_sec
    offset = int(offset_sec * self.sampling_rate_target)
    snd_shift = snd[offset:]
    return snd_shift

if __name__=="__main__":

  # prepare
  np.random.seed(0)

  # # birdclef-2022
  # split_audio_2022 = SplitAudio(
  #   dirpath_input="/kaggle/input/birdclef-2022/train_audio/",
  #   dirpath_output="/kaggle/input/birdclef-2022-modified/train_audio/"
  # )
  # df = split_audio_2022.run()
  # df.to_csv("/kaggle/input/birdclef-2022-modified/train_audio.csv", index=False)

  # # birdclef-2021
  # split_audio_2021 = SplitAudio(
  #   dirpath_input="/kaggle/input/birdclef-2021/train_short_audio/",
  #   dirpath_output="/kaggle/input/birdclef-2021-modified/train_audio/"
  # )
  # df = split_audio_2021.run()
  # df.to_csv("/kaggle/input/birdclef-2021-modified/train_audio.csv", index=False)

  # ESC-50
  split_audio_esc_50 = SplitAudio(
    dirpath_input="/workspace/data/ESC-50/audio/",
    dirpath_output="/kaggle/input/ESC-50-modified/train_audio/",
    ext="wav",
    sampling_rate_org=44100
  )
  df = split_audio_esc_50.run()
  df.to_csv("/kaggle/input/ESC-50-modified/train_audio.csv", index=False)
