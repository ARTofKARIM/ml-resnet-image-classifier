# ResNet Image Classifier

Transfer learning-based image classification using ResNet architectures (ResNet50/101/152) with fine-tuning, data augmentation, Grad-CAM visualization, and comprehensive evaluation.

## Architecture

```
ml-resnet-image-classifier/
├── src/
│   ├── data_loader.py   # Image loading with augmentation
│   ├── model.py         # ResNet builder with transfer learning
│   ├── callbacks.py     # LR schedulers, early stopping
│   ├── trainer.py       # Training orchestrator
│   ├── evaluation.py    # Metrics, confusion matrix
│   └── gradcam.py       # Grad-CAM explainability
├── config/config.yaml
├── tests/test_model.py
└── main.py
```

## Features
- Transfer learning from ImageNet-pretrained ResNet50/101/152
- Cosine annealing learning rate schedule
- Grad-CAM visualization for model interpretability
- Data augmentation pipeline

## Installation
```bash
git clone https://github.com/mouachiqab/ml-resnet-image-classifier.git
cd ml-resnet-image-classifier
pip install -r requirements.txt
```

## Usage
```bash
python main.py --data data/images/ --mode train
```

## Technologies
- Python 3.9+, TensorFlow/Keras, scikit-learn, matplotlib










