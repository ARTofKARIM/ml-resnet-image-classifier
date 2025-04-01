"""Tests for ResNet model."""
import unittest
import numpy as np
from src.model import ResNetClassifier

class TestResNetClassifier(unittest.TestCase):
    def test_build_model(self):
        clf = ResNetClassifier(num_classes=5, image_size=(32, 32), pretrained=False,
                                dense_units=[64], dropout=0.3)
        model = clf.build()
        self.assertIsNotNone(model)
        dummy = np.random.randn(2, 32, 32, 3).astype(np.float32)
        out = model.predict(dummy, verbose=0)
        self.assertEqual(out.shape, (2, 5))

    def test_output_probabilities(self):
        clf = ResNetClassifier(num_classes=3, image_size=(32, 32), pretrained=False, dense_units=[32])
        model = clf.build()
        dummy = np.random.randn(1, 32, 32, 3).astype(np.float32)
        out = model.predict(dummy, verbose=0)
        self.assertAlmostEqual(float(np.sum(out)), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
