config = {
    "random_seed": 57,
    "pred_device": "cpu",
    "label": "labels",
    "labels": [
        "abethr1", "abhori1", "abythr1", "afbfly1", "afdfly1", "afecuc1",
        "affeag1", "afgfly1", "afghor1", "afmdov1", "afpfly1", "afpkin1",
        "afpwag1", "afrgos1", "afrgrp1", "afrjac1", "afrthr1", "amesun2",
        "augbuz1", "bagwea1", "barswa", "bawhor2", "bawman1", "bcbeat1",
        "beasun2", "bkctch1", "bkfruw1", "blacra1", "blacuc1", "blakit1",
        "blaplo1", "blbpuf2", "blcapa2", "blfbus1", "blhgon1", "blhher1",
        "blksaw1", "blnmou1", "blnwea1", "bltapa1", "bltbar1", "bltori1",
        "blwlap1", "brcale1", "brcsta1", "brctch1", "brcwea1", "brican1",
        "brobab1", "broman1", "brosun1", "brrwhe3", "brtcha1", "brubru1",
        "brwwar1", "bswdov1", "btweye2", "bubwar2", "butapa1", "cabgre1",
        "carcha1", "carwoo1", "categr", "ccbeat1", "chespa1", "chewea1",
        "chibat1", "chtapa3", "chucis1", "cibwar1", "cohmar1", "colsun2",
        "combul2", "combuz1", "comsan", "crefra2", "crheag1", "crohor1",
        "darbar1", "darter3", "didcuc1", "dotbar1", "dutdov1", "easmog1",
        "eaywag1", "edcsun3", "egygoo", "equaka1", "eswdov1", "eubeat1",
        "fatrav1", "fatwid1", "fislov1", "fotdro5", "gabgos2", "gargan",
        "gbesta1", "gnbcam2", "gnhsun1", "gobbun1", "gobsta5", "gobwea1",
        "golher1", "grbcam1", "grccra1", "grecor", "greegr", "grewoo2",
        "grwpyt1", "gryapa1", "grywrw1", "gybfis1", "gycwar3", "gyhbus1",
        "gyhkin1", "gyhneg1", "gyhspa1", "gytbar1", "hadibi1", "hamerk1",
        "hartur1", "helgui", "hipbab1", "hoopoe", "huncis1", "hunsun2",
        "joygre1", "kerspa2", "klacuc1", "kvbsun1", "laudov1", "lawgol",
        "lesmaw1", "lessts1", "libeat1", "litegr", "litswi1", "litwea1",
        "loceag1", "lotcor1", "lotlap1", "luebus1", "mabeat1", "macshr1",
        "malkin1", "marsto1", "marsun2", "mcptit1", "meypar1", "moccha1",
        "mouwag1", "ndcsun2", "nobfly1", "norbro1", "norcro1", "norfis1",
        "norpuf1", "nubwoo1", "pabspa1", "palfly2", "palpri1", "piecro1",
        "piekin1", "pitwhy", "purgre2", "pygbat1", "quailf1", "ratcis1",
        "raybar1", "rbsrob1", "rebfir2", "rebhor1", "reboxp1", "reccor",
        "reccuc1", "reedov1", "refbar2", "refcro1", "reftin1", "refwar2",
        "rehblu1", "rehwea1", "reisee2", "rerswa1", "rewsta1", "rindov",
        "rocmar2", "rostur1", "ruegls1", "rufcha2", "sacibi2", "sccsun2",
        "scrcha1", "scthon1", "shesta1", "sichor1", "sincis1", "slbgre1",
        "slcbou1", "sltnig1", "sobfly1", "somgre1", "somtit4", "soucit1",
        "soufis1", "spemou2", "spepig1", "spewea1", "spfbar1", "spfwea1",
        "spmthr1", "spwlap1", "squher1", "strher", "strsee1", "stusta1",
        "subbus1", "supsta1", "tacsun1", "tafpri1", "tamdov1", "thrnig1",
        "trobou1", "varsun2", "vibsta2", "vilwea1", "vimwea1", "walsta1",
        "wbgbir1", "wbrcha2", "wbswea1", "wfbeat1", "whbcan1", "whbcou1",
        "whbcro2", "whbtit5", "whbwea1", "whbwhe3", "whcpri2", "whctur2",
        "wheslf1", "whhsaw1", "whihel1", "whrshr1", "witswa1", "wlwwar",
        "wookin1", "woosan", "wtbeat1", "yebapa1", "yebbar1", "yebduc1",
        "yebere1", "yebgre1", "yebsto1", "yeccan1", "yefcan", "yelbis1",
        "yenspu1", "yertin1", "yesbar1", "yespet1", "yetgre1", "yewgre1",
        "none"
    ],
    "experiment_name": "birdclef2023-trial-v6",
    "run_id": None,
    "path": {
        "traindata": "/kaggle/input/birdclef-2023-oversampled-5sec/train_melspec_v1/",
        "trainmeta": "/kaggle/input/birdclef-2023-oversampled-5sec/train_metadata_v1_1.csv",
        "testdata": "/kaggle/input/birdclef-2023-oversampled-5sec/train_melspec_v1/",
        "preddata": "/kaggle/input/birdclef-2023/test_soundscapes/",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/birdclef2023-trial-v6/",
        "ckpt_dir": "/workspace/tmp/checkpoint/"
    },
    "modelname": "best_loss",
    "init_with_checkpoint": False,
    "resume": False,
    "upload_every_n_epochs": 5,
    "sampling_rate": 32000,
    "chunk_sec": 5,
    "duration_sec": 5,
    "n_mels": 256,
    "pred_ensemble": False,
    "train_with_alldata": True
}
config["augmentation"] = {
    "ClassName": "SpecAugmentation",
    "time_stretch": {
        "probability": 0.1,
        "n_mels": 256
    },
    "pitch_shift": {
        "probability": 0.1,
        "max": 0.1
    },
    "time_shift": {
        "probability": 0.1,
        "max": 0.2
    },
    "freq_mask": {
        "probability": 0.1,
        "max": 25
    },
    "time_mask": {
        "probability": 0.1,
        "max": 50
    },
    "fadein": {
        "probability": 0.1,
        "max": 0.25
    },
    "fadeout": {
        "probability": 0.1,
        "max": 0.25
    }
}
config["model"] = {
    "ClassName": "BirdClefEfficientNetModel",
    "base_model_name": "tf_efficientnet_b0_ns",
    "input_channels": 1,
    "num_class": 265,
    "fc_mid_dim": 2560,
    "n_mels": config["n_mels"],
    "gradient_checkpointing": True,
    "mixup": {
        "alpha": 0.2
    },
    "label_smoothing": {
        "eps": 0.01
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
            "lr": 5e-4
        },
    },
    "scheduler":{
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params":{
            "T_0": 100,
            "eta_min": 0,
        }
    }
}
config["earlystopping"] = {
    "patience": 5
}
config["checkpoint"] = {
    "dirpath": config["path"]["model_dir"],
    "save_top_k": 1,
    "mode": "min",
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 100,
    "accumulate_grad_batches": 8,
    "deterministic": False,
    "precision": 32
}
config["datamodule"] = {
    "ClassName": "DataModule",
    "dataset":{
        "ClassName": "BirdClefMelspecDataset",
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "label": config["label"],
        "labels": config["labels"],
        "mean": 0.485,
        "std": 0.229,
        "melspec": {
            "sr": config["sampling_rate"],
            "n_mels": config["n_mels"],
            "n_fft": 2048,
            "hop_length": 512,
            "fmin": 16,
            "fmax": 16386
        },
        "path": config["path"],
        "chunk_sec": config["chunk_sec"],
        "duration_sec": config["duration_sec"]
    },
    "dataloader": {
        "batch_size": 32,
        "num_workers": 8
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
