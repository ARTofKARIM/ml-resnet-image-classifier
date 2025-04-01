"""ResNet model builder with transfer learning support."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ResNetClassifier:
    def __init__(self, num_classes, image_size=(224, 224), architecture="resnet50",
                 pretrained=True, freeze_base=True, dense_units=None, dropout=0.5):
        self.num_classes = num_classes
        self.image_size = image_size
        self.architecture = architecture
        self.pretrained = pretrained
        self.freeze_base = freeze_base
        self.dense_units = dense_units or [256, 128]
        self.dropout = dropout
        self.model = None

    def _get_base_model(self):
        input_shape = (*self.image_size, 3)
        weights = "imagenet" if self.pretrained else None
        models = {
            "resnet50": tf.keras.applications.ResNet50,
            "resnet101": tf.keras.applications.ResNet101,
            "resnet152": tf.keras.applications.ResNet152,
        }
        if self.architecture not in models:
            raise ValueError(f"Unknown arch: {self.architecture}")
        base = models[self.architecture](weights=weights, include_top=False, input_shape=input_shape)
        if self.freeze_base:
            base.trainable = False
        return base

    def build(self):
        base = self._get_base_model()
        inputs = keras.Input(shape=(*self.image_size, 3))
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        for units in self.dense_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        self.model = keras.Model(inputs, outputs)
        print(f"Model built: {self.model.count_params():,} params")
        return self.model

    def unfreeze_top_layers(self, n_layers=20):
        base = self.model.layers[3]
        base.trainable = True
        for layer in base.layers[:-n_layers]:
            layer.trainable = False
        print(f"Unfroze top {n_layers} layers for fine-tuning")

    def compile(self, learning_rate=0.001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
