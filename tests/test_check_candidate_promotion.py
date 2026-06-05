import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_summary(
    path: Path,
    *,
    bgr_delta: float = 0.03,
    ablation_delta: float = -0.01,
    radius_bgr: float = 0.60,
    radius_uniform: float = 0.55,
    seeds: int = 4,
) -> None:
    lines = ["method,seed,final_rauc,final_median_r80"]
    for seed in range(seeds):
        uniform = 0.50 + 0.001 * seed
        bgr = uniform + bgr_delta
        lines.extend(
            [
                f"uniform,{seed},{uniform:.4f},{radius_uniform:.4f}",
                f"fixed,{seed},{uniform - 0.05:.4f},0.4000",
                f"failure_only,{seed},{uniform - 0.04:.4f},0.4100",
                f"td_loss,{seed},{uniform - 0.03:.4f},0.4200",
                f"bgr_uniform_radius,{seed},{bgr + ablation_delta:.4f},0.5000",
                f"bgr,{seed},{bgr:.4f},{radius_bgr:.4f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_checker(path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "tools/check_candidate_promotion.py",
            str(path),
            "--min-seeds",
            "4",
            "--min-wins",
            "4",
            *extra_args,
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class CheckCandidatePromotionTest(unittest.TestCase):
    def test_promotable_candidate_passes_all_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_summary(path)

            result = _run_checker(path)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[decision] PROMOTABLE FOR PAPER INTEGRATION", result.stdout)

    def test_rejects_state_priority_only_ablation_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_summary(path, ablation_delta=0.01)

            result = _run_checker(path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("[decision] DO NOT PROMOTE", result.stdout)
        self.assertIn("state-priority/radius ablation", result.stdout)

    def test_rejects_ceiling_saturated_radius_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_summary(path, radius_bgr=1.0, radius_uniform=1.0)

            result = _run_checker(path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("final_median_r80 is saturated", result.stdout)

    def test_rejects_floor_saturated_radius_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_summary(path, radius_bgr=0.0, radius_uniform=0.0)

            result = _run_checker(path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("final_median_r80 is floor-saturated", result.stdout)

    def test_rejects_small_effect_against_uniform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            _write_summary(path, bgr_delta=0.005)

            result = _run_checker(path)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mean delta +0.0050 vs uniform, below +0.0100", result.stdout)


if __name__ == "__main__":
    unittest.main()
