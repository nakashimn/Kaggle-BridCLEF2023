import numpy as np
import librosa
import torch
from torchvision import transforms as Tv
from torchaudio import transforms as Ta
import traceback

class SoundAugmentation:
    def __init__(
        self,
        sampling_rate=32000,
        n_fft=2048,
        ratio_harmonic=0.0,
        ratio_pitch_shift=0.0,
        ratio_percussive=0.0,
        ratio_time_stretch=0.0,
        range_harmonic_margin=[1, 3],
        range_n_step_pitch_shift=[-0.5, 0.5],
        range_percussive_margin=[1, 3],
        range_rate_time_stretch=[0.9, 1.1]
    ):
        # const
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.enable = {
            "pitch_shift": (ratio_pitch_shift > 0.0),
            "time_stretch": (ratio_time_stretch > 0.0),
            "harmonic": (ratio_harmonic > 0.0),
            "percussive": (ratio_percussive > 0.0)
        }
        self.config = {
            "harmonic": {
                "ratio": ratio_harmonic,
                "range_margin": range_harmonic_margin
            },
            "pitch_shift": {
                "ratio": ratio_pitch_shift,
                "range_n_step": range_n_step_pitch_shift
            },
            "percussive": {
                "ratio": ratio_percussive,
                "range_margin": range_percussive_margin
            },
            "time_stretch": {
                "ratio": ratio_time_stretch,
                "range_rate": range_rate_time_stretch
            },
        }

    def __call__(self, snd):
        return self.run(snd)

    def run(self, snd):
        # check length
        if len(snd) < self.n_fft:
            return snd

        # harmonic
        if self._is_applied(self.config["harmonic"]["ratio"]):
            harmonic_margin = np.random.uniform(
                *self.config["harmonic"]["range_margin"]
            )
            snd = librosa.effects.harmonic(snd, margin=harmonic_margin)
        # pitch shift
        if self._is_applied(self.config["pitch_shift"]["ratio"]):
            pitch_shift_n_step = np.random.uniform(
                *self.config["pitch_shift"]["range_n_step"]
            )
            snd = librosa.effects.pitch_shift(
                snd, n_steps=pitch_shift_n_step, sr=self.sampling_rate
            )
        # percussive
        if self._is_applied(self.config["percussive"]["ratio"]):
            percussive_mergin = np.random.uniform(
                *self.config["percussive"]["range_margin"]
            )
            snd = librosa.effects.percussive(snd, margin=percussive_mergin)
        # time stretch
        if self._is_applied(self.config["time_stretch"]["ratio"]):
            time_stretch_rate = np.random.uniform(
                *self.config["time_stretch"]["range_rate"]
            )
            snd = librosa.effects.time_stretch(snd, rate=time_stretch_rate)
        return snd

    def _is_applied(self, ratio):
        return np.random.choice([True,False], p=[ratio, 1-ratio])


class Fadein:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, x):
        fade_length = 1.0 / (x.shape[2] * self.ratio)
        weights = torch.clip(
            fade_length * torch.arange(x.shape[2]),
            0.0, 1.0
        ).to(torch.float32)
        return x * weights

class Fadeout:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, x):
        fade_length = 1.0 / (x.shape[2] * self.ratio)
        weights = torch.flip(
            torch.clip(
                fade_length * torch.arange(x.shape[2]),
                0.0, 1.0
            ),
            dims=[0]
        ).to(torch.float32)
        return x * weights

class SpecAugmentation:
    def __init__(self, config):
        self.config = config
        self.transform = self.create_transform()
        pass

    def create_transform(self):
        augmentations = []

        ### TimeStretch
        # if self.config["time_stretch"] is not None:
        #     timestretch = Tv.RandomApply([
        #         Ta.TimeStretch(
        #             n_freq=self.config["time_stretch"]["n_mels"]
        #         )],
        #         p=self.config["time_stretch"]["probability"]
        #     )
        #     augmentations.append(timestretch)
        if self.config["pitch_shift"] is not None:
            pitchshift = Tv.RandomApply([
                Tv.RandomAffine(
                    degrees=0,
                    translate=(0, self.config["pitch_shift"]["max"])
                )],
                p=self.config["pitch_shift"]["probability"]
            )
            augmentations.append(pitchshift)
        if self.config["time_shift"] is not None:
            timeshift = Tv.RandomApply([
                Tv.RandomAffine(
                    degrees=0,
                    translate=(self.config["time_shift"]["max"], 0)
                )],
                p=self.config["time_shift"]["probability"]
            )
            augmentations.append(timeshift)
        if self.config["freq_mask"] is not None:
            freqmask = Tv.RandomApply(
                [Ta.FrequencyMasking(self.config["freq_mask"]["max"])],
                p=self.config["freq_mask"]["probability"]
            )
            augmentations.append(freqmask)
        if self.config["time_mask"] is not None:
            timemask = Tv.RandomApply(
                [Ta.TimeMasking(self.config["time_mask"]["max"])],
                p=self.config["time_mask"]["probability"]
            )
            augmentations.append(timemask)
        if self.config["fadein"] is not None:
            fadein = Tv.RandomApply(
                [Fadein(self.config["fadein"]["max"])],
                p=self.config["fadein"]["probability"]
            )
            augmentations.append(fadein)
        if self.config["fadein"] is not None:
            fadeout = Tv.RandomApply(
                [Fadeout(self.config["fadeout"]["max"])],
                p=self.config["fadeout"]["probability"]
            )
            augmentations.append(fadeout)
        if len(augmentations)==0:
            return None
        return Tv.Compose(augmentations)

    def __call__(self, melspec):
        return self.run(melspec)

    def run(self, melspec):
        return self.transform(melspec)
