import numpy as np
import librosa
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
