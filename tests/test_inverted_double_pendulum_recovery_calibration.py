import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.inverted_double_pendulum_recovery_calibration import (  # noqa: E402
    CalibrationRow,
    parse_radii,
    summarize,
)


class InvertedDoublePendulumRecoveryCalibrationTest(unittest.TestCase):
    def test_parse_radii_rejects_non_increasing_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_radii("0,0.2,0.2,0.4")

    def test_summarize_accepts_clean_nonflat_unsaturated_curve(self) -> None:
        radii = np.array([0.0, 0.3, 0.6], dtype=float)
        rows = [
            CalibrationRow(
                seed=0,
                sigma=0.0,
                angle=0.0,
                success=1,
                steps=250,
                max_abs_pole_angle=0.0,
                final_abs_pole_angle=0.0,
            ),
            CalibrationRow(
                seed=1,
                sigma=0.0,
                angle=1.0,
                success=1,
                steps=250,
                max_abs_pole_angle=0.0,
                final_abs_pole_angle=0.0,
            ),
            CalibrationRow(
                seed=0,
                sigma=0.3,
                angle=0.0,
                success=1,
                steps=250,
                max_abs_pole_angle=0.3,
                final_abs_pole_angle=0.05,
            ),
            CalibrationRow(
                seed=1,
                sigma=0.3,
                angle=1.0,
                success=0,
                steps=90,
                max_abs_pole_angle=0.5,
                final_abs_pole_angle=0.5,
            ),
            CalibrationRow(
                seed=0,
                sigma=0.6,
                angle=0.0,
                success=0,
                steps=20,
                max_abs_pole_angle=0.7,
                final_abs_pole_angle=0.7,
            ),
            CalibrationRow(
                seed=1,
                sigma=0.6,
                angle=1.0,
                success=0,
                steps=20,
                max_abs_pole_angle=0.7,
                final_abs_pole_angle=0.7,
            ),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "usable-calibration")
        self.assertEqual(summary["clean_success"], 1.0)
        self.assertEqual(summary["min_recovery"], 0.0)

    def test_summarize_rejects_flat_curve(self) -> None:
        radii = np.array([0.0, 0.3, 0.6], dtype=float)
        rows = [
            CalibrationRow(
                seed=seed,
                sigma=float(sigma),
                angle=0.0,
                success=1,
                steps=250,
                max_abs_pole_angle=float(sigma),
                final_abs_pole_angle=0.0,
            )
            for seed in (0, 1)
            for sigma in radii
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "reject-calibration-flat-recovery")


if __name__ == "__main__":
    unittest.main()
