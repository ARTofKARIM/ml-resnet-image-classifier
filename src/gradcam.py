"""Grad-CAM visualization for model interpretability."""
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv()

    def _find_last_conv(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("No conv layer found")

    def compute_heatmap(self, image, class_idx=None):
        grad_model = tf.keras.Model(self.model.inputs,
                                     [self.model.get_layer(self.layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(image)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
        grads = tape.gradient(class_output, conv_out)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        heatmap = tf.reduce_sum(conv_out * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap[0].numpy()

    def overlay(self, image, heatmap, alpha=0.4, save_path=None):
        import cv2
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]
        overlay = (1 - alpha) * image / 255.0 + alpha * heatmap_color
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title("Original")
        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[2].imshow(np.clip(overlay, 0, 1))
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
