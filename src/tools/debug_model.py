import sys
import glob
import pathlib
import torch
from torch import nn
import librosa
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperForAudioClassification
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.sample import config
from components.models import AudioClassificationModel

###
# sample
###
# base model
model_config = WhisperConfig(num_labels=10)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
base_model = WhisperForAudioClassification.from_pretrained("openai/whisper-base")
base_model.classifier = nn.Linear(256, 100, bias=True)

# prepare input
dirpath_audio = "/workspace/kaggle/input/birdclef-2023/train_audio/"
filepaths_audio = glob.glob(f"{dirpath_audio}/**/*.ogg", recursive=True)
sound, framerate = librosa.load(filepaths_audio[0], sr=44100)
sound_sr16000 = librosa.resample(sound, orig_sr=44100, target_sr=16000)
feat = feature_extractor(sound_sr16000, sampling_rate=16000, return_tensors="pt")
input_feat = feat["input_features"]
batch = torch.stack([input_feat, input_feat])

# inference
result = base_model(batch.squeeze(dim=1))
logits = result.logits
probs = logits.softmax(axis=1)
preds = probs.argmax(axis=1)

# model
model = AudioClassificationModel(config["model"])
model.decoder_input_ids
result = model(feat)
