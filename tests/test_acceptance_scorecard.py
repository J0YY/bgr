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

            text = render_markdown(root)

        self.assertIn("Grid-margin mechanism clears", text)
        self.assertIn("BGR 367/400, official 367/400", text)
        self.assertIn("already unable to clear the official-checkpoint gate", text)
        self.assertIn("pending random row is ledger completion", text)
        self.assertIn("MiniGrid LavaGapS7", text)
        self.assertIn("state-priority/uniform-radius ablation", text)
        self.assertIn("fail", text)


if __name__ == "__main__":
    unittest.main()
