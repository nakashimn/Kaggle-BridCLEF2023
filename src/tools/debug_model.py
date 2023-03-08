import sys
import pathlib
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from transformers import ViTFeatureExtractor, ViTForImageClassification
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.sample import config
from components.models import ImgRecogModel

###
# sample
###

filepath_img = "/workspace/data/img/sample_0.png"
img = read_image(filepath_img)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")


# model
model = ImgRecogModel(config["model"])

features = feature_extractor(images=img, return_tensors="pt")

result = model(**features)
logits = result.logits
