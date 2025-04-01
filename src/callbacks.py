"""Training callbacks and learning rate schedulers."""
import math
import tensorflow as tf
from tensorflow.keras import callbacks

def get_callbacks(config):
    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=config.get("early_stopping_patience", 5),
                                 restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint("models/best_model.keras", monitor="val_accuracy",
                                  save_best_only=True, mode="max"),
    ]
    return cb

class CosineAnnealingSchedule(callbacks.Callback):
    def __init__(self, initial_lr, total_epochs, min_lr=1e-7):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / self.total_epochs))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
