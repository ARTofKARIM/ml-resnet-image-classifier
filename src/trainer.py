"""Model training orchestrator."""
import yaml
from src.data_loader import ImageDataLoader
from src.model import ResNetClassifier
from src.callbacks import get_callbacks, CosineAnnealingSchedule

class Trainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model = None
        self.history = None

    def setup(self, data_dir):
        loader = ImageDataLoader(data_dir, tuple(self.config["data"]["image_size"]),
                                  self.config["data"]["batch_size"], self.config["data"]["validation_split"])
        train_ds, val_ds = loader.load_dataset()
        train_ds = loader.prepare(train_ds, augment=True)
        val_ds = loader.prepare(val_ds)
        return train_ds, val_ds, len(loader.class_names)

    def train(self, data_dir):
        train_ds, val_ds, num_classes = self.setup(data_dir)
        cfg = self.config
        classifier = ResNetClassifier(num_classes, tuple(cfg["data"]["image_size"]),
                                       cfg["model"]["architecture"], cfg["model"]["pretrained"],
                                       cfg["model"]["freeze_base"], cfg["model"]["dense_units"],
                                       cfg["model"]["dropout"])
        self.model = classifier.build()
        classifier.compile(cfg["training"]["learning_rate"])
        cb = get_callbacks(cfg["training"])
        if cfg["training"].get("lr_scheduler") == "cosine":
            cb.append(CosineAnnealingSchedule(cfg["training"]["learning_rate"], cfg["training"]["epochs"]))
        self.history = self.model.fit(train_ds, validation_data=val_ds,
                                       epochs=cfg["training"]["epochs"], callbacks=cb)
        return self.history

    def fine_tune(self, train_ds, val_ds, epochs=10, lr=0.0001):
        classifier = ResNetClassifier.__new__(ResNetClassifier)
        classifier.model = self.model
        classifier.unfreeze_top_layers(20)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)
