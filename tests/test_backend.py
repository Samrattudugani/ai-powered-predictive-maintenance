"""Integration tests for FastAPI predictive maintenance backend endpoints."""
import unittest
from fastapi.testclient import TestClient

from backend.main import app


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoints(self) -> None:
        for path in ("/health", "/api/health"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["default_subset"], "FD003")
            self.assertTrue(data["dataset_ready"])
            self.assertTrue(data["model_ready"])
            self.assertEqual(data["model_type"], "LSTM")

    def test_get_subsets(self) -> None:
        response = self.client.get("/api/subsets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["default_subset"], "FD003")
        subset_names = [item["subset"] for item in data["subsets"]]
        self.assertIn("FD001", subset_names)
        self.assertIn("FD003", subset_names)

        # FD003 should be model_ready
        fd003_info = next(item for item in data["subsets"] if item["subset"] == "FD003")
        self.assertTrue(fd003_info["model_ready"])
        self.assertTrue(fd003_info["dataset_available"])

    def test_get_metrics_success_and_validation(self) -> None:
        # Default subset (FD003)
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["subset"], "FD003")
        self.assertEqual(data["test_engines"], 100)
        self.assertGreater(data["test_mae"], 0.0)
        self.assertGreater(data["test_rmse"], 0.0)
        self.assertGreater(data["test_nasa_score"], 0.0)
        self.assertIn("best_epoch", data)

        # Explicit query param
        response_fd003 = self.client.get("/api/metrics?subset=FD003")
        self.assertEqual(response_fd003.status_code, 200)

        # Untrained subset should return 404
        response_untrained = self.client.get("/api/metrics?subset=FD002")
        self.assertEqual(response_untrained.status_code, 404)

        # Invalid subset should return 400
        response_invalid = self.client.get("/api/metrics?subset=INVALID")
        self.assertEqual(response_invalid.status_code, 400)

    def test_get_engines_list(self) -> None:
        response = self.client.get("/api/engines?subset=FD003")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["subset"], "FD003")
        self.assertEqual(data["count"], 100)
        self.assertEqual(len(data["engine_ids"]), 100)
        self.assertIsNotNone(data["attention_count"])
        self.assertGreater(data["attention_count"], 0)

        # Check first engine entry structure
        first = data["engines"][0]
        self.assertEqual(first["engine_id"], 1)
        self.assertIn("predicted_rul_cycles", first)
        self.assertIn("status", first)
        self.assertIn(first["status"], ("Normal", "Warning", "Critical"))

    def test_get_engine_history(self) -> None:
        response = self.client.get("/api/engine/1?subset=FD003")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["subset"], "FD003")
        self.assertEqual(data["engine_id"], 1)
        self.assertEqual(data["last_observed_cycle"], 233)
        self.assertEqual(data["reference_rul_cycles"], 44.0)
        self.assertEqual(data["total_cycles"], 233)
        self.assertEqual(len(data["cycles"]), 233)
        self.assertIn("s1", data["sensor_names"])

        # Engine 999 does not exist -> 404
        response_404 = self.client.get("/api/engine/999?subset=FD003")
        self.assertEqual(response_404.status_code, 404)

        # Engine ID 0 violates ge=1 -> 422
        response_422 = self.client.get("/api/engine/0?subset=FD003")
        self.assertEqual(response_422.status_code, 422)

    def test_get_engine_predict(self) -> None:
        response = self.client.get("/api/engine/1/predict?subset=FD003")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["subset"], "FD003")
        self.assertEqual(data["engine_id"], 1)
        self.assertEqual(data["last_observed_cycle"], 233)
        self.assertAlmostEqual(data["predicted_rul_cycles"], 38.82, places=1)
        self.assertEqual(data["status"], "Warning")
        self.assertEqual(data["thresholds"]["critical_cycles_or_less"], 20)
        self.assertEqual(data["thresholds"]["warning_cycles_or_less"], 50)

    def test_post_predict_with_engine_id(self) -> None:
        payload = {"subset": "FD003", "engine_id": 1}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["engine_id"], 1)
        self.assertAlmostEqual(data["predicted_rul_cycles"], 38.82, places=1)
        self.assertEqual(data["status"], "Warning")

    def test_post_predict_with_custom_sequence(self) -> None:
        # 18 features expected for FD003
        dummy_sequence = [[0.0] * 18 for _ in range(30)]
        payload = {"subset": "FD003", "sequence": dummy_sequence}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["engine_id"])
        self.assertEqual(data["last_observed_cycle"], 30)
        self.assertIsInstance(data["predicted_rul_cycles"], float)
        self.assertIn(data["status"], ("Normal", "Warning", "Critical"))

    def test_post_predict_validation_errors(self) -> None:
        # Neither engine_id nor sequence
        res_empty = self.client.post("/api/predict", json={"subset": "FD003"})
        self.assertEqual(res_empty.status_code, 400)

        # Invalid feature count (5 instead of 18)
        res_mismatch = self.client.post(
            "/api/predict", json={"subset": "FD003", "sequence": [[1.0] * 5]}
        )
        self.assertEqual(res_mismatch.status_code, 400)

        # Invalid thresholds (critical >= warning)
        res_thresh = self.client.post(
            "/api/predict",
            json={"subset": "FD003", "engine_id": 1, "critical_threshold": 60, "warning_threshold": 40},
        )
        self.assertEqual(res_thresh.status_code, 400)

    def test_untrained_subset_returns_503_for_predict(self) -> None:
        # FD002 has data but no trained model bundle
        res_get = self.client.get("/api/engine/1/predict?subset=FD002")
        self.assertEqual(res_get.status_code, 503)

        res_post = self.client.post("/api/predict", json={"subset": "FD002", "engine_id": 1})
        self.assertEqual(res_post.status_code, 503)

    def test_invalid_subset_validation_errors(self) -> None:
        res_detail = self.client.get("/api/engine/1?subset=FD999")
        self.assertEqual(res_detail.status_code, 400)

        res_pred = self.client.get("/api/engine/1/predict?subset=FD999")
        self.assertEqual(res_pred.status_code, 400)

        res_engines = self.client.get("/api/engines?subset=FD999")
        self.assertEqual(res_engines.status_code, 400)

        res_post = self.client.post("/api/predict", json={"subset": "FD999", "engine_id": 1})
        self.assertEqual(res_post.status_code, 400)

    def test_missing_engine_in_predict(self) -> None:
        res_get = self.client.get("/api/engine/999/predict?subset=FD003")
        self.assertEqual(res_get.status_code, 404)

        res_post = self.client.post("/api/predict", json={"subset": "FD003", "engine_id": 999})
        self.assertEqual(res_post.status_code, 404)

    def test_cors_headers(self) -> None:
        response = self.client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertIn("access-control-allow-origin", response.headers)



if __name__ == "__main__":
    unittest.main()
