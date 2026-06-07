import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.inverted_pendulum_recovery_calibration import CalibrationRow, parse_radii, summarize  # noqa: E402


class InvertedPendulumRecoveryCalibrationTest(unittest.TestCase):
    def test_parse_radii_rejects_non_increasing_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_radii("0,0.1,0.1,0.2")

    def test_summarize_accepts_clean_nonflat_unsaturated_curve(self) -> None:
        radii = np.array([0.0, 0.1, 0.2], dtype=float)
        rows = [
            CalibrationRow(seed=0, sigma=0.0, direction=1, success=1, steps=200, max_abs_angle=0.0, final_abs_angle=0.0),
            CalibrationRow(seed=1, sigma=0.0, direction=-1, success=1, steps=200, max_abs_angle=0.0, final_abs_angle=0.0),
            CalibrationRow(seed=0, sigma=0.1, direction=1, success=1, steps=200, max_abs_angle=0.1, final_abs_angle=0.02),
            CalibrationRow(seed=1, sigma=0.1, direction=-1, success=0, steps=40, max_abs_angle=0.2, final_abs_angle=0.2),
            CalibrationRow(seed=0, sigma=0.2, direction=1, success=0, steps=10, max_abs_angle=0.2, final_abs_angle=0.2),
            CalibrationRow(seed=1, sigma=0.2, direction=-1, success=0, steps=10, max_abs_angle=0.2, final_abs_angle=0.2),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "usable-calibration")
        self.assertEqual(summary["clean_success"], 1.0)
        self.assertEqual(summary["min_recovery"], 0.0)

    def test_summarize_rejects_flat_curve(self) -> None:
        radii = np.array([0.0, 0.1, 0.2], dtype=float)
        rows = [
            CalibrationRow(seed=seed, sigma=sigma, direction=1, success=1, steps=200, max_abs_angle=sigma, final_abs_angle=0.0)
            for seed in (0, 1)
            for sigma in radii
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "reject-calibration-flat-recovery")

    def test_summarize_rejects_low_clean_success(self) -> None:
        radii = np.array([0.0, 0.1, 0.2], dtype=float)
        rows = [
            CalibrationRow(seed=0, sigma=0.0, direction=1, success=1, steps=200, max_abs_angle=0.0, final_abs_angle=0.0),
            CalibrationRow(seed=1, sigma=0.0, direction=-1, success=0, steps=20, max_abs_angle=0.2, final_abs_angle=0.2),
            CalibrationRow(seed=0, sigma=0.1, direction=1, success=0, steps=20, max_abs_angle=0.2, final_abs_angle=0.2),
            CalibrationRow(seed=1, sigma=0.1, direction=-1, success=0, steps=20, max_abs_angle=0.2, final_abs_angle=0.2),
            CalibrationRow(seed=0, sigma=0.2, direction=1, success=0, steps=10, max_abs_angle=0.2, final_abs_angle=0.2),
            CalibrationRow(seed=1, sigma=0.2, direction=-1, success=0, steps=10, max_abs_angle=0.2, final_abs_angle=0.2),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "reject-calibration-low-clean-success")


if __name__ == "__main__":
    unittest.main()
