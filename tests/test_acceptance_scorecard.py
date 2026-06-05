import tempfile
import unittest
from pathlib import Path

from scripts.acceptance_scorecard import render_markdown
from scripts.check_acceptance_readiness import OPENVLA_WEIGHTED_AVAILABLE


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class AcceptanceScorecardTest(unittest.TestCase):
    def test_render_markdown_reports_gate_distances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "\n".join(
                    [
                        "method,seed,final_rauc",
                        "bgr,0,0.44",
                        "uniform,0,0.40",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "\n".join(
                    [
                        "method,seed,final_rauc",
                        "bgr,30,0.43",
                        "uniform,30,0.39",
                        "",
                    ]
                ),
            )
            _write(
                root / OPENVLA_WEIGHTED_AVAILABLE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,99,0.99",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,75,0.75",
                        "bgr,shift,100,95,0.95",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,99,0.99",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,99,0.99",
                        "random,occlusion,100,75,0.75",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/minigrid_lavagap_s7_recovery_probe_4seed_v1/summary.csv",
                "\n".join(
                    [
                        "method,seed,final_rauc,final_abs_r10",
                        "uniform,0,0.50,0.60",
                        "uniform,1,0.40,0.60",
                        "bgr_coverage,0,0.55,0.65",
                        "bgr_coverage,1,0.35,0.65",
                        "fixed,0,0.10,0.20",
                        "fixed,1,0.10,0.20",
                        "failure_only,0,0.30,0.50",
                        "failure_only,1,0.30,0.50",
                        "td_loss,0,0.45,0.60",
                        "td_loss,1,0.45,0.60",
                        "bgr_uniform_radius,0,0.60,0.65",
                        "bgr_uniform_radius,1,0.60,0.65",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/minigrid_fourrooms_recovery_probe_4seed_v1/summary.csv",
                "\n".join(
                    [
                        "method,seed,final_rauc,final_median_r80",
                        "uniform,0,0.10,1.00",
                        "uniform,1,0.10,1.00",
                        "uniform,2,0.10,1.00",
                        "uniform,3,0.10,1.00",
                        "fixed,0,0.12,1.00",
                        "fixed,1,0.12,1.00",
                        "fixed,2,0.12,1.00",
                        "fixed,3,0.12,1.00",
                        "failure_only,0,0.14,1.00",
                        "failure_only,1,0.14,1.00",
                        "failure_only,2,0.14,1.00",
                        "failure_only,3,0.14,1.00",
                        "td_loss,0,0.15,1.00",
                        "td_loss,1,0.15,1.00",
                        "td_loss,2,0.15,1.00",
                        "td_loss,3,0.15,1.00",
                        "bgr_uniform_radius,0,0.18,1.00",
                        "bgr_uniform_radius,1,0.18,1.00",
                        "bgr_uniform_radius,2,0.18,1.00",
                        "bgr_uniform_radius,3,0.18,1.00",
                        "bgr_coverage,0,0.22,1.00",
                        "bgr_coverage,1,0.22,1.00",
                        "bgr_coverage,2,0.22,1.00",
                        "bgr_coverage,3,0.22,1.00",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/fetchpush_object_goal_calibration_2seed_v1/summary.json",
                (
                    '{"clean_success": 0.25, "min_recovery": 0.25, '
                    '"max_recovery": 0.25, "r80": 0.12}\n'
                ),
            )
            _write(
                root / "results/fetchslide_object_goal_calibration_2seed_v1/summary.json",
                (
                    '{"clean_success": 0.90, "min_recovery": 0.40, '
                    '"max_recovery": 0.90, "r80": 0.09}\n'
                ),
            )
            _write(
                root / "results/fetchpickplace_object_goal_calibration_2seed_v1/summary.json",
                (
                    '{"clean_success": 0.25, "min_recovery": 0.25, '
                    '"max_recovery": 0.25, "r80": 0.15}\n'
                ),
            )
            _write(
                root / "AGENTS.md",
                (
                    "The proximal-anchor OpenVLA route is queued with "
                    "BGR jobs 767128--767143 and random jobs 767144--767148.\n"
                ),
            )

            text = render_markdown(root)

        self.assertIn("Grid-margin mechanism clears", text)
        self.assertIn("BGR 367/400, official 367/400", text)
        self.assertIn("already unable to clear the official-checkpoint gate", text)
        self.assertIn("pending random row is ledger completion", text)
        self.assertIn("Proximal-anchor OpenVLA route is in flight, not yet evidence", text)
        self.assertIn("must finish before the +10/400 and +0.02 learned-policy gate", text)
        self.assertIn("MiniGrid LavaGapS7", text)
        self.assertIn("MiniGrid FourRooms official-package", text)
        self.assertIn("final_median_r80-ceiling-saturated", text)
        self.assertIn("Cleared gates", text)
        self.assertIn("3/4", text)
        self.assertIn("state-priority/uniform-radius ablation", text)
        self.assertIn("Pre-Method Calibrations", text)
        self.assertIn("FetchPush-v4 object-goal calibration", text)
        self.assertIn("FetchSlide-v4 object-goal calibration", text)
        self.assertIn("FetchPickAndPlace-v4 object-goal calibration", text)
        self.assertIn("reject-calibration", text)
        self.assertIn("usable-calibration", text)
        self.assertIn("fail", text)
        self.assertIn("queued proximal-anchor OpenVLA route", text)
        self.assertIn("Do not start another same-protocol MiniGrid", text)


if __name__ == "__main__":
    unittest.main()
