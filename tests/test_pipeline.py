"""
Unit tests for the ML pipeline
"""

import os
import sys
import tempfile
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.feature_engineering import load_data
from src.preprocessing import preprocess_image


class TestPipeline(unittest.TestCase):
    def test_preprocess_blank_image_returns_none(self):
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess_image(image_array=blank_image)
        self.assertIsNone(result)

    def test_load_data_empty_directory_returns_empty_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            X, y = load_data(data_dir=temp_dir)
            self.assertEqual(X.shape, (0,))
            self.assertEqual(y.shape, (0,))


if __name__ == "__main__":
    unittest.main()
