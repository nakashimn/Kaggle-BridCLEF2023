import os
import subprocess
from pytorch_lightning import callbacks
import traceback

class ModelUploader(callbacks.Callback):
    def __init__(self, model_dir, every_n_epochs=5, message=""):
        self.model_dir = model_dir
        self.every_n_epochs = every_n_epochs
        self.message = message
        self.should_upload = False
        super().__init__()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if self.should_upload:
            self._upload_model(
                f"{self.message}[epoch:{trainer.current_epoch}]"
            )
            self.should_upload = False
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self.every_n_epochs is None:
            return super().on_train_epoch_end(trainer, pl_module)
        if (self.every_n_epochs <= 0):
            return super().on_train_epoch_end(trainer, pl_module)
        if (trainer.current_epoch == 0):
            return super().on_train_epoch_end(trainer, pl_module)
        if (trainer.current_epoch % self.every_n_epochs == 0):
            self.should_upload = True
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        if self.every_n_epochs is not None:
            self._upload_model(
                f"{self.message}[epoch:{trainer.current_epoch}]"
            )
        return super().on_train_end(trainer, pl_module)

    def _upload_model(self, message):
        try:
            subprocess.run(
                ["kaggle", "datasets", "version", "-m", message],
                cwd=self.model_dir
            )
        except:
            print(traceback.format_exc())
            raise
