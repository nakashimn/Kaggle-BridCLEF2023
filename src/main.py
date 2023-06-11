import os
import shutil
import importlib
import argparse
import random
import gc
import subprocess
import pathlib
import glob
import numpy as np
import torch
import traceback

from components.preprocessor import DataPreprocessor
from components.trainer import Trainer

def update_config(config, filepath_config):
    # copy ConfigFile from temporal_dir to model_dir
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    filename_config = pathlib.Path(filepath_config).name
    shutil.copy2(filepath_config, str(dirpath_model / filename_config))

def remove_exist_models(config):
    filepaths_ckpt = glob.glob(f"{config['path']['model_dir']}/*.ckpt")
    for fp in filepaths_ckpt:
        os.remove(fp)

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_classes(config):
    # import Classes dynamically
    Model = getattr(
        importlib.import_module(f"components.models"),
        config["model"]["ClassName"]
    )
    Dataset = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["dataset"]["ClassName"]
    )
    DataModule = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["ClassName"]
    )
    Augmentation = getattr(
        importlib.import_module(f"components.augmentation"),
        config["augmentation"]["ClassName"]
    )
    # debug message
    print("================================================")
    print(f"Components:")
    print(f"  Model        : {Model.__name__}")
    print(f"  Dataset      : {Dataset.__name__}")
    print(f"  DataModule   : {DataModule.__name__}")
    print(f"  Augmentation : {Augmentation.__name__}")
    print("================================================")
    return Model, Dataset, DataModule, Augmentation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help="stem of config filepath.",
        type=str,
        required=True
    )
    return parser.parse_args()


if __name__=="__main__":

    # args
    args = get_args()
    config = importlib.import_module(f"config.{args.config}").config

    # import Classes
    Model, Dataset, DataModule, Augmentation = import_classes(config)

    # update config
    update_config(config, f"./config/{args.config}.py")

    # create model dir
    os.makedirs(config["path"]["model_dir"], exist_ok=True)

    # torch setting
    torch.set_float32_matmul_precision("medium")

    # Preprocess
    remove_exist_models(config)
    fix_seed(config["random_seed"])
    data_preprocessor = DataPreprocessor(config)
    df_train = data_preprocessor.train_dataset_primary()

    # Training
    trainer = Trainer(
        Model,
        DataModule,
        Dataset,
        Augmentation,
        df_train,
        config
    )
    trainer.run()
