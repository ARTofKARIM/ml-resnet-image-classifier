"""Model evaluation and analysis."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir

    def evaluate(self, model, dataset, class_names):
        y_true, y_pred = [], []
        for images, labels in dataset:
            preds = model.predict(images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(labels.numpy())
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        accuracy = np.mean(y_true == y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))
        return {"accuracy": accuracy, "report": report, "y_true": y_true, "y_pred": y_pred}

    def plot_confusion_matrix(self, y_true, y_pred, class_names, save=True):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")
        if save:
            fig.savefig(f"{self.output_dir}confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_training_history(self, history, save=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history.history["loss"], label="Train")
        ax1.plot(history.history.get("val_loss", []), label="Val")
        ax1.set_title("Loss")
        ax1.legend()
        ax2.plot(history.history["accuracy"], label="Train")
        ax2.plot(history.history.get("val_accuracy", []), label="Val")
        ax2.set_title("Accuracy")
        ax2.legend()
        if save:
            fig.savefig(f"{self.output_dir}training_history.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
