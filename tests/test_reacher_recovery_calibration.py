import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.reacher_recovery_calibration import (  # noqa: E402
    CalibrationRow,
    LINK_1,
    LINK_2,
    ik_solutions,
    parse_radii,
    summarize,
    target_joint_angles,
)


class ReacherRecoveryCalibrationTest(unittest.TestCase):
    def test_parse_radii_rejects_non_increasing_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_radii("0,0.5,0.5,1.0")

    def test_ik_solutions_reach_target(self) -> None:
        target = np.array([0.12, 0.05], dtype=float)

        for q1, q2 in ik_solutions(target):
            fingertip = np.array(
                [
                    LINK_1 * np.cos(q1) + LINK_2 * np.cos(q1 + q2),
                    LINK_1 * np.sin(q1) + LINK_2 * np.sin(q1 + q2),
                ],
                dtype=float,
            )
            self.assertLess(float(np.linalg.norm(fingertip - target)), 1e-9)

    def test_target_joint_angles_prefers_nearest_branch(self) -> None:
        target = np.array([0.10, 0.10], dtype=float)
        solutions = ik_solutions(target)
        current = solutions[1] + np.array([0.01, -0.01], dtype=float)

        selected = target_joint_angles(target, current)

        np.testing.assert_allclose(selected, solutions[1])

    def test_summarize_accepts_clean_nonflat_curve(self) -> None:
        radii = np.array([0.0, 1.0, 2.0], dtype=float)
        rows = [
            CalibrationRow(seed=0, sigma=0.0, success=1, steps=16, best_distance=0.01, final_distance=0.02),
            CalibrationRow(seed=1, sigma=0.0, success=1, steps=16, best_distance=0.01, final_distance=0.02),
            CalibrationRow(seed=0, sigma=1.0, success=1, steps=16, best_distance=0.02, final_distance=0.03),
            CalibrationRow(seed=1, sigma=1.0, success=0, steps=16, best_distance=0.05, final_distance=0.06),
            CalibrationRow(seed=0, sigma=2.0, success=0, steps=16, best_distance=0.08, final_distance=0.09),
            CalibrationRow(seed=1, sigma=2.0, success=0, steps=16, best_distance=0.07, final_distance=0.08),
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "usable-calibration")
        self.assertEqual(summary["clean_success"], 1.0)
        self.assertEqual(summary["min_recovery"], 0.0)

    def test_summarize_rejects_flat_curve(self) -> None:
        radii = np.array([0.0, 1.0, 2.0], dtype=float)
        rows = [
            CalibrationRow(seed=seed, sigma=sigma, success=1, steps=16, best_distance=0.01, final_distance=0.02)
            for seed in (0, 1)
            for sigma in radii
        ]

        summary = summarize(rows, radii, alpha=0.80)

        self.assertEqual(summary["decision"], "reject-calibration-flat-recovery")


if __name__ == "__main__":
    unittest.main()
