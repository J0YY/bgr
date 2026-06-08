import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.highway_lane_recovery_calibration import CalibrationRow, parse_radii, summarize


class HighwayLaneRecoveryCalibrationTest(unittest.TestCase):
    def test_parse_radii_rejects_non_increasing_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_radii("0,1,0.5")

    def test_summarize_rejects_saturated_radius(self) -> None:
        radii = np.array([0.0, 1.0, 2.0], dtype=float)
        rows = []
        for seed in range(5):
            rows.append(CalibrationRow(seed=seed, sigma=0.0, success=1, crashed=0, on_road=1, steps=40))
            rows.append(CalibrationRow(seed=seed, sigma=1.0, success=1, crashed=0, on_road=1, steps=40))
            rows.append(
                CalibrationRow(
                    seed=seed,
                    sigma=2.0,
                    success=0 if seed == 0 else 1,
                    crashed=1 if seed == 0 else 0,
                    on_road=1,
                    steps=12 if seed == 0 else 40,
                )
            )

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "reject-calibration-saturated-radius")
        self.assertEqual(summary["clean_success"], 1.0)

    def test_summarize_accepts_nonflat_clean_curve(self) -> None:
        radii = np.array([0.0, 1.0, 2.0], dtype=float)
        rows = [
            CalibrationRow(seed=0, sigma=0.0, success=1, crashed=0, on_road=1, steps=40),
            CalibrationRow(seed=1, sigma=0.0, success=1, crashed=0, on_road=1, steps=40),
            CalibrationRow(seed=0, sigma=1.0, success=1, crashed=0, on_road=1, steps=40),
            CalibrationRow(seed=1, sigma=1.0, success=0, crashed=1, on_road=1, steps=18),
            CalibrationRow(seed=0, sigma=2.0, success=0, crashed=1, on_road=1, steps=10),
            CalibrationRow(seed=1, sigma=2.0, success=0, crashed=1, on_road=1, steps=9),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "usable-calibration")
        self.assertEqual(summary["min_recovery"], 0.0)
        self.assertEqual(summary["max_recovery"], 1.0)
        self.assertEqual(summary["mean_crash_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
