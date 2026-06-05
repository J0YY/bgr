import tempfile
import unittest
from pathlib import Path

from scripts.check_acceptance_readiness import OPENVLA_WEIGHTED_AVAILABLE
from scripts.check_acceptance_readiness import independent_benchmark_gate
from scripts.check_acceptance_readiness import learned_policy_gate
from scripts.check_acceptance_readiness import roadmap_hygiene_gate


def _write_weighted_summary(root: Path) -> None:
    path = root / OPENVLA_WEIGHTED_AVAILABLE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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
        encoding="utf-8",
    )


def _write_independent_summaries(root: Path) -> None:
    summaries = {
        "results/frozenlake_recovery_focused_30seed_v1/summary.csv": (
            "method,seed,final_rauc\n"
            "bgr,0,0.50\nuniform,0,0.40\nfailure_only,0,0.60\n"
        ),
        "results/minigrid_fourrooms_recovery_probe_midband_4seed_v1/summary.csv": (
            "method,seed,final_rauc\n"
            "bgr_coverage,0,0.50\nuniform,0,0.60\nfailure_only,0,0.70\n"
        ),
        "results/minigrid_doorkey_recovery_probe_4seed_v1/summary.csv": (
            "method,seed,final_rauc\n"
            "bgr_coverage,0,0.50\nuniform,0,0.60\nfailure_only,0,0.70\n"
        ),
        "results/minigrid_lavacrossing_recovery_probe_4seed_v1/summary.csv": (
            "method,seed,final_rauc\n"
            "bgr_coverage,0,0.50\nuniform,0,0.60\nbgr_uniform_radius,0,0.70\n"
        ),
        "results/minigrid_lavagap_s7_recovery_probe_4seed_v1/summary.csv": (
            "method,seed,final_rauc\n"
            "bgr_coverage,0,0.50\nuniform,0,0.60\nbgr_uniform_radius,0,0.70\n"
        ),
        "results/pointmaze_umaze_clean_shield_probe_4seed_v1/summary.csv": (
            "method,seed,final_rauc\n"
            "bgr_clean_shield,0,0.50\nuniform,0,0.40\nfailure_only,0,0.70\n"
        ),
        "results/fetchreach_goal_recovery_probe_4seed_v1/summary.csv": (
            "method,seed,final_rauc,final_median_r80\n"
            "bgr_coverage,0,0.50,0.15\nbgr,0,0.49,0.15\nuniform,0,0.60,0.15\nfailure_only,0,0.70,0.15\n"
        ),
    }
    for relative, text in summaries.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


class CheckAcceptanceReadinessTest(unittest.TestCase):
    def test_learned_policy_gate_prefers_latest_weighted_official_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_weighted_summary(root)

            gate = learned_policy_gate(root)

        self.assertFalse(gate.passed)
        self.assertEqual(gate.name, "learned-policy OpenVLA/LIBERO")
        self.assertIn("latest weighted audit", gate.detail)
        self.assertIn("BGR 367/400", gate.detail)
        self.assertIn("official 367/400", gate.detail)
        self.assertIn("random 273/300 available rows", gate.detail)
        self.assertIn("official_margin=0", gate.detail)
        self.assertIn("official gate already impossible", gate.detail)

    def test_independent_gate_reports_fetchpush_invalid_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_independent_summaries(root)
            path = root / "results/fetchpush_object_goal_calibration_2seed_v1/summary.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                '{"clean_success": 0.25, "min_recovery": 0.25, "max_recovery": 0.25, "r80": 0.12}\n',
                encoding="utf-8",
            )
            slide_path = root / "results/fetchslide_object_goal_calibration_2seed_v1/summary.json"
            slide_path.parent.mkdir(parents=True, exist_ok=True)
            slide_path.write_text(
                '{"clean_success": 0.25, "min_recovery": 0.25, "max_recovery": 0.25, "r80": 0.15}\n',
                encoding="utf-8",
            )
            pick_path = root / "results/fetchpickplace_object_goal_calibration_2seed_v1/summary.json"
            pick_path.parent.mkdir(parents=True, exist_ok=True)
            pick_path.write_text(
                '{"clean_success": 0.25, "min_recovery": 0.25, "max_recovery": 0.25, "r80": 0.15}\n',
                encoding="utf-8",
            )

            gate = independent_benchmark_gate(root)

        self.assertFalse(gate.passed)
        self.assertIn("FetchPush calibration invalid", gate.detail)
        self.assertIn("FetchSlide calibration invalid", gate.detail)
        self.assertIn("FetchPickAndPlace calibration invalid", gate.detail)
        self.assertIn("clean 0.2500", gate.detail)

    def test_roadmap_hygiene_rejects_completed_weighted_openvla_as_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir(parents=True)
            (root / "AGENTS.md").write_text("Current status.\n", encoding="utf-8")
            (docs / "review_weakness_response.md").write_text("Current status.\n", encoding="utf-8")
            (docs / "aaai_acceptance_gap.md").write_text(
                "The next preregistered learned-policy intervention is weighted perturbation curriculum.\n",
                encoding="utf-8",
            )

            gate = roadmap_hygiene_gate(root)

        self.assertFalse(gate.passed)
        self.assertIn("weighted perturbation", gate.detail)

    def test_roadmap_hygiene_accepts_completed_negative_weighted_openvla(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir(parents=True)
            (root / "AGENTS.md").write_text(
                "The latest learned-policy follow-up is the preregistered weighted OpenVLA perturbation curriculum. "
                "It is a negative audit.\n",
                encoding="utf-8",
            )
            (docs / "aaai_acceptance_gap.md").write_text(
                "The latest preregistered learned-policy intervention is the weighted perturbation curriculum. "
                "This intervention is now a negative audit for the official-checkpoint gate.\n",
                encoding="utf-8",
            )
            (docs / "review_weakness_response.md").write_text(
                "Do not keep rerunning the same MiniGrid/PointMaze/FetchReach protocol family.\n",
                encoding="utf-8",
            )

            gate = roadmap_hygiene_gate(root)

        self.assertTrue(gate.passed)
        self.assertIn("active roadmap avoids stale next-step instructions", gate.detail)


if __name__ == "__main__":
    unittest.main()
