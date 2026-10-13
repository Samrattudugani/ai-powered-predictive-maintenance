import unittest
import numpy as np

from src.train import nasa_score


class EvaluationTests(unittest.TestCase):
    def test_nasa_score_zero_error(self) -> None:
        actual = np.array([50.0, 30.0, 10.0])
        predicted = np.array([50.0, 30.0, 10.0])
        # When error = 0, exp(0) - 1 = 0
        score = nasa_score(predicted, actual)
        self.assertAlmostEqual(score, 0.0)

    def test_nasa_score_asymmetric_penalty(self) -> None:
        # Optimistic (late) error: d = +10 -> exp(10/10) - 1 = e - 1 ~ 1.71828
        late_pred = np.array([30.0])
        actual = np.array([20.0])
        late_score = nasa_score(late_pred, actual)
        expected_late = np.exp(1.0) - 1.0
        self.assertAlmostEqual(late_score, expected_late, places=4)

        # Conservative (early) error: d = -10 -> exp(10/13) - 1 ~ 1.15818
        early_pred = np.array([10.0])
        early_score = nasa_score(early_pred, actual)
        expected_early = np.exp(10.0 / 13.0) - 1.0
        self.assertAlmostEqual(early_score, expected_early, places=4)

        # Optimistic error should have a higher penalty than pessimistic error of the same magnitude
        self.assertGreater(late_score, early_score)


if __name__ == "__main__":
    unittest.main()
