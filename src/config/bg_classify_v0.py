config = {
    "n_splits": 5,
    "train_fold": [0, 1, 2, 3, 4],
    "valid_fold": [0, 1, 2, 3, 4],
    "random_seed": 57,
    "pred_device": "gpu",
    "label": "labels",
    "labels": ["True", "False"],
    "group": "group",
    "experiment_name": "birdclef2023-bg-classify-v0",
    "path": {
        "traindata": "/kaggle/input/birdclef-2023-background-5sec/train_audio/",
        "trainmeta": "/kaggle/input/birdclef-2023-background-5sec/train_metadata.csv",
        "testdata": "/kaggle/input/birdclef-2023-background-5sec/train_audio/",
        "preddata": "/kaggle/input/birdclef-2023-background-5sec/test_soundscapes/",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/birdclef2023-bg-classify-v0/"
    },
    "modelname": "best_loss",
    "sampling_rate": 32000,
    "chunk_sec": 5,
    "duration_sec": 5,
    "pred_ensemble": False,
    "train_with_alldata": True
}
config["augmentation"] = {
    "sampling_rate": config["sampling_rate"],
    "ratio_harmonic": 0.10,
    "ratio_pitch_shift": 0.10,
    "ratio_percussive": 0.10,
    "ratio_time_stretch": 0.0,
    "range_harmonic_margin": [1, 3],
    "range_n_step_pitch_shift": [-0.5, 0.5],
    "range_percussive_margin": [1, 3],
    "range_rate_time_stretch": [0.9, 1.1]
}
config["model"] = {
    "base_model_name": "/workspace/data/model/birdclef2023_pretrained/",
    "fc_feature_dim": 2048,
    "num_class": 2,
    "gradient_checkpointing": True,
    "freeze_base_model": False,
    "model_config": {
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 2048
    },
    "loss": {
        "name": "nn.CrossEntropyLoss",
        "params": {
            "weight": None
        }
    },
    "optimizer":{
        "name": "optim.RAdam",
        "params":{
            "lr": 1e-4
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
    "patience": 2
}
config["checkpoint"] = {
    "dirpath": config["path"]["temporal_dir"],
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
    "deterministic": False,
    "num_sanity_val_steps": 0,
    "precision": 32
}
config["kfold"] = {
    "name": "StratifiedKFold",
    "params": {
        "n_splits": config["n_splits"],
        "shuffle": True,
        "random_state": config["random_seed"]
    },
    "anchor": {
    }
}
config["datamodule"] = {
    "dataset":{
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "label": config["label"],
        "labels": config["labels"],
        "sampling_rate": {
            "org": config["sampling_rate"],
            "target": 16000
        },
        "path": config["path"],
        "chunk_sec": config["chunk_sec"],
        "duration_sec": config["duration_sec"],
        "max_length": 80000
    },
    "train_loader": {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
        "drop_last": False
    },
    "pred_loader": {
        "batch_size": 16,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": False,
        "drop_last": False
    }
}
config["Metrics"] = {
    "confmat": {
        "label": config["labels"]
    },
    "cmAP": {
        "padding_num": 5
    }
}
