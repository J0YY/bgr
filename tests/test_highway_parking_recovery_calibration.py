import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.highway_parking_recovery_calibration import CalibrationRow, parse_radii, summarize


class HighwayParkingRecoveryCalibrationTest(unittest.TestCase):
    def test_parse_radii_rejects_non_increasing_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_radii("0,1,1,2")

    def test_summarize_rejects_low_clean_success(self) -> None:
        radii = np.array([0.0, 1.0, 2.0], dtype=float)
        rows = [
            CalibrationRow(seed=0, sigma=0.0, success=0, crashed=1, steps=10),
            CalibrationRow(seed=1, sigma=0.0, success=1, crashed=0, steps=10),
            CalibrationRow(seed=0, sigma=1.0, success=0, crashed=1, steps=10),
            CalibrationRow(seed=1, sigma=1.0, success=1, crashed=0, steps=10),
            CalibrationRow(seed=0, sigma=2.0, success=0, crashed=1, steps=10),
            CalibrationRow(seed=1, sigma=2.0, success=0, crashed=1, steps=10),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "reject-calibration-low-clean-success")
        self.assertEqual(summary["clean_success"], 0.5)
        self.assertEqual(summary["seeds"], 2)

    def test_summarize_accepts_nonflat_clean_curve(self) -> None:
        radii = np.array([0.0, 1.0, 2.0], dtype=float)
        rows = [
            CalibrationRow(seed=0, sigma=0.0, success=1, crashed=0, steps=5),
            CalibrationRow(seed=1, sigma=0.0, success=1, crashed=0, steps=5),
            CalibrationRow(seed=0, sigma=1.0, success=1, crashed=0, steps=5),
            CalibrationRow(seed=1, sigma=1.0, success=0, crashed=0, steps=5),
            CalibrationRow(seed=0, sigma=2.0, success=0, crashed=0, steps=5),
            CalibrationRow(seed=1, sigma=2.0, success=0, crashed=0, steps=5),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "usable-calibration")
        self.assertEqual(summary["min_recovery"], 0.0)
        self.assertEqual(summary["max_recovery"], 1.0)


if __name__ == "__main__":
    unittest.main()
