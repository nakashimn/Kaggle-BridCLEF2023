import sys
import glob
import pathlib
import torch
from torch import nn
import librosa
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model, PretrainedConfig
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.trial_v4 import config
from components.models import BirdClefModel, BirdClefPretrainModel, TimmSEDBaseModel, BirdClefTimmSEDSimSiamModel

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
model_ = BirdClefModel(config["model"])
result = model(input_feat)

# pretrained model_v0
from config.pretrain_v0 import config
model = BirdClefPretrainModel.load_from_checkpoint(
    f"{config['path']['model_dir']}/{config['modelname']}.ckpt",
    config=config["model"]
)
model.model = model.model.wav2vec2
model.save_pretrained_model()

from config.pretrain_v2 import config
model =BirdClefTimmSEDSimSiamModel.load_from_checkpoint(
    f"{config['path']['model_dir']}/{config['modelname']}.ckpt",
    config=config["model"]
)
model.save_pretrained_model()

dummy_config = PretrainedConfig()
TimmSEDBaseModel.from_pretrained("/workspace/data/model/birdclef2023_pretrained_v1/", config=dummy_config)
