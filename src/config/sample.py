config = {
    "n_splits": 2,
    "train_fold": [0, 1],
    "valid_fold": [0, 1],
    "random_seed": 57,
    "label": "class",
    "group": "group",
    "labels": [
        "A",
        "B"
    ],
    "experiment_name": "sample-v0",
    "path": {
        "traindata": "/workspace/data/sample/labels.csv",
        "testdata": "/workspace/data/sample/labels.csv",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/workspace/data/model/sample/"
    },
    "modelname": "best_loss",
    "pred_ensemble": True,
    "train_with_alldata": False
}
config["model"] = {
    "base_model_name": "WinKawaks/vit-tiny-patch16-224",
    "dim_feature": 1000,
    "num_class": 2,
    "dropout_rate": 0.5,
    "freeze_base_model": False,
    "loss": {
        "name": "nn.CrossEntropyLoss",
        "params": {
            "weight": None
        }
    },
    "optimizer":{
        "name": "optim.RAdam",
        "params":{
            "lr": 1e-5
        },
    },
    "scheduler":{
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params":{
            "T_0": 20,
            "eta_min": 1e-4,
        }
    }
}
config["earlystopping"] = {
    "patience": 1
}
config["checkpoint"] = {
    "dirpath": config["path"]["temporal_dir"],
    "monitor": "val_loss",
    "save_top_k": 1,
    "mode": "min",
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 100,
    "accumulate_grad_batches": 1,
    "fast_dev_run": False,
    "deterministic": True,
    "num_sanity_val_steps": 0,
    "resume_from_checkpoint": None,
    "precision": 16
}
config["kfold"] = {
    "name": "StratifiedGroupKFold",
    "params": {
        "n_splits": config["n_splits"],
        "shuffle": True,
        "random_state": config["random_seed"]
    }
}
config["datamodule"] = {
    "dataset":{
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "label": config["label"],
        "labels": config["labels"],
        "use_fast_tokenizer": True,
        "max_length": 512
    },
    "train_loader": {
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": False
    },
    "pred_loader": {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": False,
        "drop_last": False
    }
}
config["Metrics"] = {
    "label": config["labels"]
}
