import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from src.inference import determine_health_status, predict_rul, load_model_bundle
from src.model import RULLSTM


class InferenceTests(unittest.TestCase):
    def test_determine_health_status(self) -> None:
        # Default thresholds: Critical <= 20, Warning <= 50, Normal > 50
        self.assertEqual(determine_health_status(10.0), "Critical")
        self.assertEqual(determine_health_status(20.0), "Critical")
        self.assertEqual(determine_health_status(20.1), "Warning")
        self.assertEqual(determine_health_status(50.0), "Warning")
        self.assertEqual(determine_health_status(50.5), "Normal")
        self.assertEqual(determine_health_status(120.0), "Normal")

    def test_predict_rul_shape_and_clipping(self) -> None:
        model = RULLSTM(input_size=4, hidden_size=8, num_layers=1)
        model.eval()

        # Batch of 5 sequences
        sequences = np.random.randn(5, 10, 4).astype(np.float32)
        preds = predict_rul(sequences, model, clip_min=0.0)

        self.assertEqual(preds.shape, (5,))
        self.assertTrue((preds >= 0.0).all())

    def test_predict_rul_invalid_dimensions(self) -> None:
        model = RULLSTM(input_size=4, hidden_size=8, num_layers=1)
        with self.assertRaises(ValueError):
            predict_rul(np.random.randn(10, 4), model)

    def test_load_saved_artifact_bundle_fd003(self) -> None:
        from src.data import ROOT
        artifacts_dir = ROOT / "artifacts"
        if not (artifacts_dir / "model_fd003.pt").is_file():
            self.skipTest("FD003 model artifacts not yet generated.")

        model, scaler, config = load_model_bundle("FD003", artifacts_dir=artifacts_dir)
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertEqual(config["subset"], "FD003")
        self.assertEqual(len(config["features"]), 18)
        self.assertEqual(config["sequence_length"], 30)

        # Predict with a valid dummy sequence
        dummy_seq = np.zeros((1, 30, 18), dtype=np.float32)
        pred = predict_rul(dummy_seq, model, clip_min=0.0)
        self.assertEqual(pred.shape, (1,))
        self.assertGreaterEqual(float(pred[0]), 0.0)


if __name__ == "__main__":

    unittest.main()
