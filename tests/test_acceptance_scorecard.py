import tempfile
import unittest
from pathlib import Path

from scripts.acceptance_scorecard import render_markdown
from scripts.check_acceptance_readiness import OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE
from scripts.check_acceptance_readiness import OPENVLA_PROXIMAL_ANCHOR_COMPLETE
from scripts.check_acceptance_readiness import OPENVLA_WEIGHTED_COMPLETE


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class AcceptanceScorecardTest(unittest.TestCase):
    def test_render_markdown_reports_perturb_only_anchor_inflight_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )
            _write(
                root / "docs/aaai_acceptance_gap.md",
                "scripts/queue_openvla_oft_preregistered_perturb_only_anchor.sh is preregistered.\n",
            )

            text = render_markdown(root)

        self.assertIn("Proximal-anchor OpenVLA audit does not clear", text)
        self.assertIn("Perturb-only anchored OpenVLA route is preregistered, not yet evidence", text)
        self.assertIn("Active route: Perturb-only anchored OpenVLA route is preregistered", text)

    def test_render_markdown_prefers_completed_perturb_only_anchor_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )
            _write(
                root / OPENVLA_PERTURB_ONLY_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,99,0.99",
                        "bgr,blur,100,99,0.99",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,85,0.85",
                        "bgr,shift,100,99,0.99",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,99,0.99",
                        "random,blur,100,97,0.97",
                        "random,brightness,100,97,0.97",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,95,0.95",
                        "",
                    ]
                ),
            )

            text = render_markdown(root)

        self.assertIn("Perturb-only anchored OpenVLA audit clears", text)
        self.assertIn("BGR 382/400, official 367/400, random 362/400", text)
        self.assertNotIn("Proximal-anchor OpenVLA audit", text)

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
                root / OPENVLA_WEIGHTED_COMPLETE,
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
                        "random,shift,100,97,0.97",
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
                root / "results/highway_parking_recovery_calibration_12seed_v1/summary.json",
                (
                    '{"clean_success": 0.3333333333, "min_recovery": 0.25, '
                    '"max_recovery": 0.50, "r80": 9.8}\n'
                ),
            )
            _write(
                root / "AGENTS.md",
                (
                    "The proximal-anchor OpenVLA route is queued with "
                    "BGR jobs 767657--767678 and random jobs 767681--767685.\n"
                ),
            )

            text = render_markdown(root)

        self.assertIn("Grid-margin mechanism clears", text)
        self.assertIn("BGR 367/400, official 367/400", text)
        self.assertIn("complete and fails the learned-policy gate", text)
        self.assertIn("trails matched random by 3/400", text)
        self.assertIn("Proximal-anchor OpenVLA route is unsummarized, not yet evidence", text)
        self.assertIn("must finish before the +10/400 and +0.02 learned-policy gate", text)
        self.assertIn("Promotion Deficits", text)
        self.assertIn("no screen clears the 4/4 promotion screen", text)
        self.assertIn("short of the best-comparator promotion gate by 13/400", text)
        self.assertIn("Proximal-anchor OpenVLA route is unsummarized", text)
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
        self.assertIn("highway-env parking-v0 calibration", text)
        self.assertIn("reject-calibration", text)
        self.assertIn("usable-calibration", text)
        self.assertIn("fail", text)
        self.assertIn("unsummarized, not yet evidence", text)
        self.assertIn("Do not start another same-protocol MiniGrid", text)

    def test_render_markdown_prefers_completed_proximal_anchor_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_WEIGHTED_COMPLETE,
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
                        "random,shift,100,97,0.97",
                        "",
                    ]
                ),
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,99,0.99",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,75,0.75",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,88,0.88",
                        "random,identity,100,99,0.99",
                        "random,blur,100,95,0.95",
                        "random,brightness,100,96,0.96",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,88,0.88",
                        "",
                    ]
                ),
            )

            text = render_markdown(root)

        self.assertIn("Proximal-anchor OpenVLA audit clears", text)
        self.assertIn("BGR 370/400, official 357/400, random 352/400", text)
        self.assertIn("official gap +13", text)
        self.assertIn("latest summarized OpenVLA audit clears", text)
        self.assertNotIn("Weighted OpenVLA audit", text)
        self.assertNotIn("latest weighted OpenVLA audit is short", text)
        self.assertNotIn("in flight, not yet evidence", text)

    def test_priority_read_prefers_completed_proximal_anchor_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_WEIGHTED_COMPLETE,
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
                        "random,shift,100,97,0.97",
                        "",
                    ]
                ),
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )

            text = render_markdown(root)

        priority_read = text.split("## Priority Read", maxsplit=1)[1]
        self.assertIn("latest proximal-anchor OpenVLA audit is complete and fails", priority_read)
        self.assertIn("BGR 368/400, official 367/400, random 368/400", priority_read)
        self.assertNotIn("latest OpenVLA weighted audit", priority_read)

    def test_retired_calibrated_routes_are_not_reported_as_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/reacher_recovery_calibration_12seed_v1/summary.json",
                '{"clean_success": 0.8333, "min_recovery": 0.5, "max_recovery": 0.9167, "r80": 3.0}\n',
            )
            _write(
                root / "results/reacher_recovery_probe_12seed_v1/summary.csv",
                "method,seed,final_rauc,final_median_r80\nuniform,0,0.38,3.0\nbgr,0,0.29,3.0\n",
            )
            _write(
                root / "results/inverted_pendulum_recovery_calibration_12seed_v1/summary.json",
                '{"clean_success": 1.0, "min_recovery": 0.0, "max_recovery": 1.0, "r80": 0.21}\n',
            )
            _write(
                root / "results/inverted_pendulum_recovery_probe_4seed_v1/summary.csv",
                "method,seed,final_rauc,final_median_r80\nuniform,0,0.75,0.21\nbgr,0,0.75,0.21\n",
            )

            text = render_markdown(root)

        self.assertIn("Retired calibrated route(s): `Gymnasium MuJoCo Reacher-v5 calibration`, `Gymnasium MuJoCo InvertedPendulum-v5 calibration`", text)
        self.assertIn("not active acceptance evidence", text)
        self.assertIn("Active route: no queued learned-policy route is recorded", text)
        self.assertNotIn("all corresponding completed method screens are negative or absent", text)

    def test_active_inverted_double_pendulum_calibration_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/inverted_double_pendulum_recovery_calibration_12seed_v1/summary.json",
                '{"clean_success": 1.0, "min_recovery": 0.0, "max_recovery": 1.0, "r80": 0.2825}\n',
            )

            text = render_markdown(root)

        self.assertIn("Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration", text)
        self.assertIn("Active route: `Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration`", text)
        self.assertIn("fixed all-method screen for the active pre-method calibration route", text)
        self.assertNotIn("Retired calibrated route(s): `Gymnasium MuJoCo InvertedDoublePendulum-v5 calibration`", text)

    def test_render_markdown_reports_rejected_route_scout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/sklearn_digits_margin_scout_v0/summary.csv",
                "\n".join(
                    [
                        "target_radius,method,n,final_rauc_mean,delta_vs_uniform,wins_vs_uniform,losses_vs_uniform,ties_vs_uniform,decision",
                        "0.8000,uniform,4,0.812250,0.000000,0,0,4,reject-scout",
                        "0.8000,fixed,4,0.842500,0.030250,3,1,0,reject-scout",
                        "0.8000,bgr,4,0.691125,-0.121125,1,3,0,reject-scout",
                        "1.0000,uniform,4,0.812250,0.000000,0,0,4,reject-scout",
                        "1.0000,fixed,4,0.793375,-0.018875,1,3,0,reject-scout",
                        "1.0000,bgr,4,0.827063,0.014813,2,2,0,reject-scout",
                        "",
                    ]
                ),
            )

            text = render_markdown(root)

        self.assertIn("Rejected route scout(s): `sklearn digits margin replay`", text)
        self.assertIn("Route Scouts", text)
        self.assertIn("0.8271", text)
        self.assertIn("+0.0148 (2/2/0)", text)
        self.assertIn("0.8425 @ 0.8000", text)
        self.assertIn("reject-scout", text)
        self.assertIn("Most pre-existing-dataset route scouts are rejected", text)

    def test_render_markdown_reports_multi_dataset_route_scout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/sklearn_tabular_margin_scout_v0/summary.csv",
                "\n".join(
                    [
                        "dataset,target_radius,method,n,final_rauc_mean,delta_vs_uniform,wins_vs_uniform,losses_vs_uniform,ties_vs_uniform,decision",
                        "breast_cancer,2.0000,uniform,4,0.951641,0.000000,0,0,4,reject-scout",
                        "breast_cancer,2.0000,fixed,4,0.955469,0.003828,2,2,0,reject-scout",
                        "breast_cancer,2.0000,bgr,4,0.961016,0.009375,3,1,0,reject-scout",
                        "wine,0.5000,uniform,4,0.956349,0.000000,0,0,4,reject-scout",
                        "wine,0.5000,fixed,4,0.958581,0.002232,2,2,0,reject-scout",
                        "wine,0.5000,bgr,4,0.970238,0.013889,4,0,0,reject-scout",
                        "",
                    ]
                ),
            )

            text = render_markdown(root)

        self.assertIn("sklearn tabular margin replay (breast_cancer)", text)
        self.assertIn("sklearn tabular margin replay (wine)", text)
        self.assertIn("+0.0094 (3/1/0)", text)
        self.assertIn("+0.0139 (4/0/0)", text)
        self.assertIn("reject-scout", text)

    def test_render_markdown_reports_promotable_openml_route_scout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/openml_margin_scout_v0/summary.csv",
                "\n".join(
                    [
                        "dataset,target_radius,method,n,final_rauc_mean,delta_vs_uniform,wins_vs_uniform,losses_vs_uniform,ties_vs_uniform,decision",
                        "diabetes,2.0000,uniform,4,0.679687,0.000000,0,0,4,reject-scout",
                        "diabetes,2.0000,fixed,4,0.699937,0.020250,4,0,0,reject-scout",
                        "diabetes,2.0000,bgr,4,0.740187,0.060500,4,0,0,candidate-for-preregistration",
                        "sonar,1.5000,uniform,4,0.737800,0.000000,0,0,4,reject-scout",
                        "sonar,1.5000,fixed,4,0.749144,0.011344,2,2,0,reject-scout",
                        "sonar,1.5000,bgr,4,0.766695,0.028896,4,0,0,reject-scout",
                        "",
                    ]
                ),
            )

            text = render_markdown(root)

        self.assertIn("OpenML margin replay (diabetes)", text)
        self.assertIn("Route scout(s): `OpenML margin replay (diabetes)` need a fixed preregistered comparison", text)
        self.assertIn("+0.0605 (4/0/0)", text)
        self.assertIn("0.6999 @ 2.0000", text)
        self.assertIn("candidate-for-preregistration", text)

    def test_render_markdown_reports_positive_openml_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root / "results/grid_margin_full_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,0,0.44\nuniform,0,0.40\n",
            )
            _write(
                root / "results/grid_margin_full_replication_30seed_v1/summary.csv",
                "method,seed,final_rauc\nbgr,30,0.43\nuniform,30,0.39\n",
            )
            _write(
                root / OPENVLA_PROXIMAL_ANCHOR_COMPLETE,
                "\n".join(
                    [
                        "method,perturbation,episodes,successes,success_rate",
                        "bgr,identity,100,98,0.98",
                        "bgr,blur,100,98,0.98",
                        "bgr,brightness,100,99,0.99",
                        "bgr,occlusion,100,73,0.73",
                        "bgr,shift,100,98,0.98",
                        "official,identity,100,99,0.99",
                        "official,blur,100,97,0.97",
                        "official,brightness,100,98,0.98",
                        "official,occlusion,100,74,0.74",
                        "official,shift,100,98,0.98",
                        "random,identity,100,98,0.98",
                        "random,blur,100,99,0.99",
                        "random,brightness,100,98,0.98",
                        "random,occlusion,100,73,0.73",
                        "random,shift,100,98,0.98",
                        "",
                    ]
                ),
            )
            _write(
                root / "results/openml_diabetes_margin_30seed_v1/summary.csv",
                "\n".join(
                    [
                        "dataset,target_radius,method,n,final_rauc_mean,delta_vs_uniform,wins_vs_uniform,losses_vs_uniform,ties_vs_uniform,decision",
                        "diabetes,2.0000,uniform,30,0.668892,0.000000,0,0,30,reject-scout",
                        "diabetes,2.0000,fixed,30,0.675917,0.007025,16,14,0,reject-scout",
                        "diabetes,2.0000,bgr,30,0.706192,0.037300,24,6,0,reject-scout",
                        "",
                    ]
                ),
            )

            text = render_markdown(root)

        self.assertIn("Positive pre-existing-dataset follow-up(s): `OpenML diabetes margin 30-seed (diabetes)`", text)
        self.assertIn("fixed gap +0.0303", text)
        self.assertIn("positive-follow-up", text)


if __name__ == "__main__":
    unittest.main()
