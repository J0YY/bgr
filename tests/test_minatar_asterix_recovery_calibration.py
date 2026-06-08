import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.minatar_asterix_recovery_calibration import CalibrationRow, parse_radii, summarize


class MinAtarAsterixRecoveryCalibrationTest(unittest.TestCase):
    def test_parse_radii_rejects_fractional_offsets(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer player-cell"):
            parse_radii("0,0.5,1")

    def test_summarize_accepts_clean_nonflat_unsaturated_curve(self) -> None:
        radii = np.array([0.0, 1.0, 2.0], dtype=float)
        rows = [
            CalibrationRow(0, 0.0, 1, 0, 60, 2, 5, 5, 5, 5),
            CalibrationRow(1, 0.0, 1, 0, 60, 2, 5, 5, 5, 5),
            CalibrationRow(0, 1.0, 1, 0, 60, 2, 5, 5, 4, 5),
            CalibrationRow(1, 1.0, 0, 1, 25, 0, 5, 5, 6, 5),
            CalibrationRow(0, 2.0, 0, 1, 20, 0, 5, 5, 3, 6),
            CalibrationRow(1, 2.0, 0, 1, 18, 0, 5, 5, 7, 4),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "usable-calibration")
        self.assertEqual(summary["clean_success"], 1.0)
        self.assertEqual(summary["min_recovery"], 0.0)
        self.assertEqual(summary["mean_terminal_rate"], 0.5)

    def test_summarize_rejects_saturated_radius(self) -> None:
        radii = np.array([0.0, 1.0, 2.0], dtype=float)
        rows = [
            CalibrationRow(seed, sigma, 1, 0, 60, 2, 5, 5, 5, 5)
            for seed in range(3)
            for sigma in radii
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "reject-calibration-flat-recovery")


if __name__ == "__main__":
    unittest.main()
