import unittest

import numpy as np

from tools.handreach_recovery_calibration import CalibrationRow
from tools.handreach_recovery_calibration import active_fingers_and_dims
from tools.handreach_recovery_calibration import parse_radii
from tools.handreach_recovery_calibration import summarize


class HandReachRecoveryCalibrationTest(unittest.TestCase):
    def test_active_fingers_include_thumb_for_pair_goals(self) -> None:
        obs = {
            "achieved_goal": np.zeros(15, dtype=float),
            "desired_goal": np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.06,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.05,
                    0.0,
                ],
                dtype=float,
            ),
        }

        fingers, dims = active_fingers_and_dims(obs, threshold=0.015)

        self.assertEqual(fingers, [1, 4])
        self.assertIn(5, dims)
        self.assertIn(19, dims)
        self.assertIn(0, dims)
        self.assertIn(1, dims)

    def test_parse_radii_rejects_unsorted_grid(self) -> None:
        with self.assertRaises(ValueError):
            parse_radii("0.0,0.2,0.1")

    def test_summarize_rejects_low_clean_success(self) -> None:
        radii = np.array([0.0, 0.1], dtype=float)
        rows = [
            CalibrationRow(0, 0.0, 0, 0.05, 0.08, "1,4", "0,1,5,6,7,15,16,17,18,19"),
            CalibrationRow(1, 0.0, 1, 0.00, 0.00, "", "0,1"),
            CalibrationRow(0, 0.1, 0, 0.06, 0.09, "1,4", "0,1,5,6,7,15,16,17,18,19"),
            CalibrationRow(1, 0.1, 1, 0.01, 0.01, "", "0,1"),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["clean_success"], 0.5)
        self.assertEqual(summary["decision"], "reject-calibration-low-clean-success")


if __name__ == "__main__":
    unittest.main()
