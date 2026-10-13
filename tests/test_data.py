import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import data


def trajectory_row(engine_id: int, cycle: int) -> str:
    values = [engine_id, cycle, cycle / 10, cycle / 20, 100]
    values.extend(cycle + sensor / 10 for sensor in range(1, 22))
    return " ".join(str(value) for value in values)


class DataPipelineTests(unittest.TestCase):
    def test_load_subset_parses_whitespace_sorts_and_aligns_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "train_FD003.txt").write_text(
                f"{trajectory_row(2, 2)}  \n{trajectory_row(1, 2)}\n{trajectory_row(1, 1)}\n",
                encoding="utf-8",
            )
            (directory / "test_FD003.txt").write_text(
                f"{trajectory_row(2, 3)}\n{trajectory_row(1, 1)}\n", encoding="utf-8"
            )
            (directory / "RUL_FD003.txt").write_text("17 \n29\n", encoding="utf-8")
            with patch.object(data, "DATA_DIR", directory):
                train, test, rul = data.load_subset()

        self.assertEqual(train.shape, (3, 26))
        self.assertEqual(train[["engine_id", "cycle"]].values.tolist(), [[1, 1], [1, 2], [2, 2]])
        self.assertEqual(test[["engine_id", "cycle"]].values.tolist(), [[1, 1], [2, 3]])
        self.assertEqual(rul.tolist(), [17.0, 29.0])

    def test_missing_required_files_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            data, "DATA_DIR", Path(temporary)
        ):
            with self.assertRaisesRegex(FileNotFoundError, "train_FD003.txt"):
                data.load_subset()

    def test_training_targets_are_clipped_and_short_windows_are_left_padded(self) -> None:
        frame = pd.DataFrame(
            {"engine_id": [1, 1, 1], "cycle": [1, 2, 3], "s1": [1.0, 2.0, 3.0]}
        )
        labeled = data.add_train_targets(frame, rul_cap=1)
        np.testing.assert_array_equal(labeled["rul"].to_numpy(), [1, 1, 0])
        window = data.make_window(np.array([[2.0], [3.0]]), window_size=4)
        np.testing.assert_array_equal(window[:, 0], [2.0, 2.0, 2.0, 3.0])

    def test_scaler_fit_uses_only_provided_training_rows(self) -> None:
        training_rows = pd.DataFrame(
            {"setting_1": [0.0, 2.0], "setting_2": [100.0, 100.0], "s1": [5.0, 7.0]}
        )
        scaler, features = data.fit_scaler(training_rows)
        self.assertEqual(features, ["setting_1", "s1"])
        np.testing.assert_allclose(scaler.mean_, [1.0, 6.0])

    def test_unsupported_subset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported subset"):
            data.load_subset("FD005")


if __name__ == "__main__":
    unittest.main()
