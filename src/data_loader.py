"""Image data loading with augmentation pipeline."""
import tensorflow as tf
import os

class ImageDataLoader:
    def __init__(self, data_dir, image_size=(224, 224), batch_size=32, val_split=0.2):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.class_names = None

    def load_dataset(self):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir, validation_split=self.val_split, subset="training",
            seed=42, image_size=self.image_size, batch_size=self.batch_size,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir, validation_split=self.val_split, subset="validation",
            seed=42, image_size=self.image_size, batch_size=self.batch_size,
        )
        self.class_names = train_ds.class_names
        print(f"Classes: {self.class_names}")
        print(f"Train batches: {tf.data.experimental.cardinality(train_ds)}")
        return train_ds, val_ds

    def get_augmentation_layer(self):
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])

    def prepare(self, dataset, augment=False):
        AUTOTUNE = tf.data.AUTOTUNE
        if augment:
            aug = self.get_augmentation_layer()
            dataset = dataset.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        return dataset.prefetch(buffer_size=AUTOTUNE)
