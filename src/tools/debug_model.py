import sys
import glob
import pathlib
import torch
from torch import nn
import librosa
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.trial_v3 import config
from components.models import BirdClefModel

###
# sample
###
# base model
model_config = Wav2Vec2Config(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)
feature_extractor = Wav2Vec2FeatureExtractor()
base_model = Wav2Vec2Model(model_config)
linear = nn.Linear(768, 265)

# prepare input
dirpath_audio = "/workspace/kaggle/input/birdclef-2023/train_audio/"
filepaths_audio = glob.glob(f"{dirpath_audio}/**/*.ogg", recursive=True)
duration_sec = 30
sound, framerate = librosa.load(filepaths_audio[0], sr=32000, duration=duration_sec)
sound_sr16000 = librosa.resample(sound, orig_sr=32000, target_sr=16000)
feat = feature_extractor(
    sound_sr16000,
    max_length=480000,
    padding="max_length",
    truncation=True,
    sampling_rate=16000,
    return_tensors="pt"
)
input_feat = feat["input_values"]
batch = torch.stack([input_feat, input_feat])

# inference
out = base_model(batch.squeeze(dim=1))
logits = linear(out.last_hidden_state[:, 0])
probs = logits.softmax(axis=1)
preds = probs.argmax(axis=1)

# model
model = BirdClefModel(config["model"])
result = model(input_feat)
