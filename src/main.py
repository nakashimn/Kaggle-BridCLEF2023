import os
import shutil
import importlib
import argparse
import pathlib
import glob
import torch
import traceback

from components.utility import print_info, fix_seed
from components.preprocessor import DataPreprocessor
from components.trainer import Trainer

def update_config(
        config: dict,
        filepath_config: str
    ):
    # copy ConfigFile from temporal_dir to model_dir
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    if not os.path.exists(dirpath_model):
        print("\033[31m" + "Warning: No Model Dir." + "\033[0m")
    filename_config = pathlib.Path(filepath_config).name
    try:
        shutil.copy2(filepath_config, str(dirpath_model / filename_config))
    except FileNotFoundError:
        raise


def remove_exist_models(
        config: dict
    ):
    filepaths_ckpt = glob.glob(f"{config['path']['model_dir']}/*.ckpt")
    for fp in filepaths_ckpt:
        os.remove(fp)

def import_classes(
        config: dict
    ):
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
        importlib.import_module(f"components.augmentations"),
        config["augmentation"]["ClassName"]
    )
    # debug message
    print_info(
        {
            "Components": {
                "Model": Model.__name__,
                "Dataset": Dataset.__name__,
                "DataModule": DataModule.__name__,
                "Augmentation": Augmentation.__name__
            }
        }
    )
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
