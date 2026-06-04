import csv
import tempfile
import unittest
from pathlib import Path

from scripts.check_paper_claims import (
    Claim,
    find_significance_row,
    forbidden_terms,
    forbidden_paper_negative_claims,
    fmt,
    mean_metric,
    mean_success_rate,
    missing_claims,
    pooled_success_rate,
    ratio,
    unverified_result_claims,
    validate_grid_margin_full_30_significance,
    validate_suffix_coverage_full_significance,
    validate_significance_checks,
)


class CheckPaperClaimsTest(unittest.TestCase):
    def test_mean_metric_filters_method_rows(self):
        rows = [
            {"method": "bgr", "final_rauc": "0.4"},
            {"method": "bgr", "final_rauc": "0.6"},
            {"method": "uniform", "final_rauc": "0.3"},
        ]

        self.assertEqual(mean_metric(rows, "bgr", "final_rauc"), 0.5)

    def test_format_helpers_match_paper_style(self):
        self.assertEqual(fmt(0.4345018, 4), "0.4345")
        self.assertEqual(fmt(0.4345018, 3), "0.435")
        self.assertEqual(ratio("14", "15"), "14/15")

    def test_missing_claims_reports_absent_snippets(self):
        claims = [
            Claim("present", "0.4345 vs. 0.3961", "summary.csv"),
            Claim("missing", "0.4961 vs. 0.4844", "summary.csv"),
        ]

        missing = missing_claims("BGR improves 0.4345 vs. 0.3961.", claims)

        self.assertEqual([claim.label for claim in missing], ["missing"])

    def test_forbidden_terms_rejects_old_method_name(self):
        self.assertEqual(forbidden_terms("Boundary-Guided Replay"), [])
        self.assertEqual(forbidden_terms("Bifurcation-Guided Replay"), ["Bifurcation"])

    def test_forbidden_terms_rejects_openvla_table_framing(self):
        text = r"\input{figures/openvla_table.tex} \label{tab:openvla} Table~\ref{tab:openvla}"

        self.assertEqual(
            forbidden_terms(text),
            [
                r"\input{figures/openvla_table.tex}",
                r"\label{tab:openvla}",
                r"Table~\ref{tab:openvla}",
            ],
        )

    def test_forbidden_paper_negative_claims_rejects_p4096_mentions(self):
        self.assertEqual(forbidden_paper_negative_claims("OpenVLA p2048 audit only."), [])
        self.assertEqual(
            forbidden_paper_negative_claims("The p4096 common-availability diagnostic is negative."),
            ["p4096", "common-availability"],
        )

    def test_unverified_result_claims_rejects_queued_p1024_without_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = unverified_result_claims("The cleanmix_p1024 result improves robustness.", Path(temp_dir))

        self.assertEqual(len(missing), 1)
        self.assertIn("cleanmix_p1024", missing[0])

    def test_unverified_result_claims_rejects_current_p1024_wording_without_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = unverified_result_claims(
                "The p1024 clean-mix run is current; the latest p1024 diagnostic improves robustness.",
                Path(temp_dir),
            )

        self.assertEqual(len(missing), 2)
        self.assertIn("p1024 clean-mix", missing[0])
        self.assertIn("latest p1024 diagnostic", missing[1])

    def test_unverified_result_claims_rejects_p1024_offset_without_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = unverified_result_claims("The offset-3 follow-up pooled result is 0.8550.", Path(temp_dir))

        self.assertEqual(len(missing), 2)
        self.assertIn("offset-3 follow-up", missing[0])
        self.assertIn("0.8550", missing[1])

    def test_unverified_result_claims_rejects_p2048_without_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = unverified_result_claims("A p2048 scale-up shows that p2048 ties official.", Path(temp_dir))

        self.assertEqual(len(missing), 2)
        self.assertIn("p2048 scale-up", missing[0])
        self.assertIn("p2048 ties", missing[1])

    def test_unverified_result_claims_accepts_p1024_after_local_summaries_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            for path in [
                results_dir
                / "openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                results_dir
                / "openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                results_dir
                / "openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")

            missing = unverified_result_claims("The latest p1024 clean-mix diagnostic and offset-3 follow-up improve robustness.", results_dir)

        self.assertEqual(missing, [])

    def test_unverified_result_claims_accepts_p2048_after_local_summaries_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            for path in [
                results_dir
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                results_dir
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")

            missing = unverified_result_claims("A p2048 scale-up shows that p2048 ties official.", results_dir)

        self.assertEqual(missing, [])

    def test_mean_success_rate_can_exclude_identity(self):
        rows = [
            {"method": "bgr", "perturbation": "identity", "success_rate": "1.0"},
            {"method": "bgr", "perturbation": "blur", "success_rate": "0.8"},
            {"method": "bgr", "perturbation": "shift", "success_rate": "0.6"},
            {"method": "random", "perturbation": "blur", "success_rate": "0.5"},
        ]

        self.assertEqual(mean_success_rate(rows, "bgr", exclude_perturbations={"identity"}), 0.7)

    def test_pooled_success_rate_weights_by_episode_count(self):
        first = [
            {"method": "bgr", "perturbation": "blur", "episodes": "10", "successes": "8"},
            {"method": "random", "perturbation": "blur", "episodes": "10", "successes": "7"},
        ]
        second = [
            {"method": "bgr", "perturbation": "blur", "episodes": "30", "successes": "27"},
            {"method": "bgr", "perturbation": "identity", "episodes": "30", "successes": "30"},
        ]

        self.assertEqual(pooled_success_rate([first, second], "bgr", exclude_perturbations={"identity"}), 35 / 40)

    def test_find_significance_row_requires_unique_match(self):
        rows = [
            {
                "benchmark": "bench",
                "condition": "",
                "metric": "final_rauc",
                "treatment": "bgr",
                "baseline": "uniform",
            }
        ]

        row = find_significance_row(
            rows,
            benchmark="bench",
            condition="",
            metric="final_rauc",
            treatment="bgr",
            baseline="uniform",
        )

        self.assertEqual(row["metric"], "final_rauc")
        with self.assertRaises(ValueError):
            find_significance_row(
                rows,
                benchmark="bench",
                condition="missing",
                metric="final_rauc",
                treatment="bgr",
                baseline="uniform",
            )

    def test_validate_significance_checks_rejects_flipped_claim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            figures_dir = Path(temp_dir)
            write_significance_csv(figures_dir / "significance_tests.csv", supports_grid=False)

            with self.assertRaisesRegex(ValueError, "grid 15-seed diagnostic final RAUC sign test"):
                validate_significance_checks(figures_dir)

    def test_suffix_coverage_full_significance_accepts_complete_30seed_rows(self):
        rows = suffix_coverage_full_rows()

        self.assertEqual(
            validate_suffix_coverage_full_significance(rows),
            ["Robot suffix coverage-full significance rows use complete positive original and held-out 30-seed comparisons"],
        )

    def test_suffix_coverage_full_significance_rejects_incomplete_rows(self):
        rows = suffix_coverage_full_rows()[:-1]

        with self.assertRaisesRegex(ValueError, "significance rows mismatch"):
            validate_suffix_coverage_full_significance(rows)

    def test_suffix_coverage_full_significance_rejects_unsupported_rows(self):
        rows = suffix_coverage_full_rows()
        rows[0]["supports_treatment"] = "false"
        rows[0]["paired_wins"] = "29"
        rows[0]["paired_losses"] = "1"

        with self.assertRaisesRegex(ValueError, "must support BGR-Coverage"):
            validate_suffix_coverage_full_significance(rows)

    def test_grid_margin_full_30_significance_accepts_complete_rows(self):
        rows = grid_margin_full_30_rows()

        self.assertEqual(
            validate_grid_margin_full_30_significance(rows),
            ["Grid margin full 30-seed significance rows use complete positive 30-seed comparisons"],
        )

    def test_grid_margin_full_30_significance_rejects_unsupported_rows(self):
        rows = grid_margin_full_30_rows()
        rows[0]["paired_wins"] = "29"
        rows[0]["paired_losses"] = "1"

        with self.assertRaisesRegex(ValueError, "must support BGR"):
            validate_grid_margin_full_30_significance(rows)

def write_significance_csv(path: Path, *, supports_grid: bool) -> None:
    rows = []
    for benchmark, condition, metric, treatment, baseline, supports, wins, losses in expected_rows():
        if benchmark == "Grid margin 15-seed" and metric == "final_rauc":
            supports = supports_grid
        n = str(wins + losses)
        rows.append(
            {
                "benchmark": benchmark,
                "condition": condition,
                "metric": metric,
                "treatment": treatment,
                "baseline": baseline,
                "n": n,
                "mean_treatment_minus_baseline": "0.1" if supports else "-0.1",
                "paired_se": "0.001",
                "paired_ci95_low": "0.09" if supports else "-0.11",
                "paired_ci95_high": "0.11" if supports else "-0.09",
                "paired_wins": str(wins),
                "paired_losses": str(losses),
                "paired_ties": "0",
                "two_sided_sign_test_p": "0.0001",
                "direction": "higher",
                "supports_treatment": str(supports).lower(),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def suffix_coverage_full_rows() -> list[dict[str, str]]:
    rows = []
    for benchmark in ["Robot suffix coverage-full 30-seed", "Robot suffix coverage-full replication 30-seed"]:
        for metric, baseline in [
            ("final_rauc", "clean_ft"),
            ("final_rauc", "fixed"),
            ("final_rauc", "failure_only"),
            ("final_rauc", "loss_priority"),
            ("final_rauc", "uniform"),
            ("final_transfer_rauc", "uniform"),
            ("rauc_aulc", "uniform"),
        ]:
            rows.append(
                {
                    "benchmark": benchmark,
                    "condition": "",
                    "metric": metric,
                    "treatment": "bgr_broad",
                    "baseline": baseline,
                    "n": "30",
                    "mean_treatment_minus_baseline": "0.1",
                    "paired_se": "0.001",
                    "paired_ci95_low": "0.09",
                    "paired_ci95_high": "0.11",
                    "paired_wins": "30",
                    "paired_losses": "0",
                    "paired_ties": "0",
                    "two_sided_sign_test_p": "0.0000",
                    "direction": "higher",
                    "supports_treatment": "true",
                }
            )
    return rows


def grid_margin_full_30_rows() -> list[dict[str, str]]:
    rows = []
    for metric, baseline in [
        ("final_rauc", "uniform"),
        ("rauc_aulc", "uniform"),
        ("final_clean", "uniform"),
        ("final_median_r80", "uniform"),
        ("final_rauc", "fixed"),
        ("final_rauc", "failure_only"),
        ("final_rauc", "plr_loss"),
        ("rauc_aulc", "fixed"),
        ("rauc_aulc", "failure_only"),
        ("rauc_aulc", "plr_loss"),
    ]:
        rows.append(
            {
                "benchmark": "Grid margin full 30-seed",
                "condition": "",
                "metric": metric,
                "treatment": "bgr",
                "baseline": baseline,
                "n": "30",
                "mean_treatment_minus_baseline": "0.1",
                "paired_se": "0.001",
                "paired_ci95_low": "0.09",
                "paired_ci95_high": "0.11",
                "paired_wins": "30",
                "paired_losses": "0",
                "paired_ties": "0",
                "two_sided_sign_test_p": "1.86265e-09",
                "direction": "higher",
                "supports_treatment": "true",
            }
        )
    return rows


def expected_rows() -> list[tuple[str, str, str, str, str, bool, int, int]]:
    rows = [
        ("Synthetic margin 30-seed", "", "final_rauc", "bgr", "uniform", True, 29, 1),
        ("Synthetic margin 30-seed", "", "rauc_aulc", "bgr", "uniform", True, 30, 0),
        ("Synthetic margin 30-seed", "", "final_clean", "bgr", "uniform", True, 30, 0),
        ("Synthetic margin 30-seed", "", "final_rauc", "bgr", "fixed", True, 30, 0),
        ("Synthetic margin 30-seed", "", "final_rauc", "bgr", "failure_only", True, 30, 0),
        ("Synthetic margin 30-seed", "", "final_rauc", "bgr", "plr_loss", True, 30, 0),
        ("Grid margin 15-seed", "", "final_rauc", "bgr", "uniform", True, 15, 0),
        ("Grid margin full 15-seed", "", "final_rauc", "bgr", "fixed", True, 15, 0),
        ("Grid margin full 15-seed", "", "final_rauc", "bgr", "failure_only", True, 15, 0),
        ("Grid margin full 15-seed", "", "final_rauc", "bgr", "plr_loss", True, 15, 0),
        ("Grid margin full 30-seed", "", "final_rauc", "bgr", "uniform", True, 30, 0),
        ("Grid margin full 30-seed", "", "rauc_aulc", "bgr", "uniform", True, 30, 0),
        ("Grid margin full 30-seed", "", "final_clean", "bgr", "uniform", True, 30, 0),
        ("Grid margin full 30-seed", "", "final_median_r80", "bgr", "uniform", True, 30, 0),
        ("Grid margin full 30-seed", "", "final_rauc", "bgr", "fixed", True, 30, 0),
        ("Grid margin full 30-seed", "", "final_rauc", "bgr", "failure_only", True, 30, 0),
        ("Grid margin full 30-seed", "", "final_rauc", "bgr", "plr_loss", True, 30, 0),
        ("Grid margin full 30-seed", "", "rauc_aulc", "bgr", "fixed", True, 30, 0),
        ("Grid margin full 30-seed", "", "rauc_aulc", "bgr", "failure_only", True, 30, 0),
        ("Grid margin full 30-seed", "", "rauc_aulc", "bgr", "plr_loss", True, 30, 0),
        ("Grid margin replication 30-seed", "", "final_rauc", "bgr", "uniform", True, 30, 0),
        ("Grid margin replication 30-seed", "", "rauc_aulc", "bgr", "uniform", True, 30, 0),
        ("Grid margin replication 30-seed", "", "final_clean", "bgr", "uniform", True, 30, 0),
        ("Grid margin ablation 15-seed", "", "final_rauc", "bgr", "bgr_uniform_radius", True, 15, 0),
        ("Grid margin ablation 15-seed", "", "final_rauc", "bgr_uniform_radius", "uniform", False, 0, 15),
        ("Robot suffix coverage 30-seed", "", "final_clean", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage 30-seed", "", "final_transfer_rauc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage 30-seed", "", "rauc_aulc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage 30-seed", "", "final_median_r80", "bgr_broad", "uniform", False, 1, 29),
        ("Robot suffix replication 30-seed", "", "final_clean", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix replication 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix replication 30-seed", "", "final_transfer_rauc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix replication 30-seed", "", "rauc_aulc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "clean_ft", True, 30, 0),
        ("Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "fixed", True, 30, 0),
        ("Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "failure_only", True, 30, 0),
        ("Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "loss_priority", True, 30, 0),
        ("Robot suffix coverage-full 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage-full 30-seed", "", "final_transfer_rauc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage-full 30-seed", "", "rauc_aulc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "clean_ft", True, 30, 0),
        ("Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "fixed", True, 30, 0),
        ("Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "failure_only", True, 30, 0),
        ("Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "loss_priority", True, 30, 0),
        ("Robot suffix coverage-full replication 30-seed", "", "final_rauc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage-full replication 30-seed", "", "final_transfer_rauc", "bgr_broad", "uniform", True, 30, 0),
        ("Robot suffix coverage-full replication 30-seed", "", "rauc_aulc", "bgr_broad", "uniform", True, 30, 0),
        ("Estimator 15-seed", "", "boundary_hit_rate", "active", "uniform", True, 15, 0),
        ("Estimator 30-seed", "", "boundary_hit_rate", "active", "uniform", True, 30, 0),
        ("Estimator 30-seed", "", "r80_mae", "active", "uniform", True, 30, 0),
        ("Estimator 30-seed", "", "rauc_mae", "active", "uniform", True, 24, 6),
    ]
    for target_margin in [0.26, 0.32, 0.38, 0.46, 0.54]:
        for metric in ["final_rauc", "rauc_aulc"]:
            rows.append(
                (
                    "Grid margin target sensitivity 15-seed",
                    f"target_margin={target_margin:.2f}",
                    metric,
                    f"bgr_target_{target_margin:.2f}",
                    "uniform",
                    True,
                    15,
                    0,
                )
            )
        for metric in ["final_rauc", "rauc_aulc", "final_clean"]:
            rows.append(
                (
                    "Grid margin target sensitivity 30-seed",
                    f"target_margin={target_margin:.2f}",
                    metric,
                    f"bgr_target_{target_margin:.2f}",
                    "uniform",
                    True,
                    30,
                    0,
                )
            )
    for regime in ["high_obstacle", "low_obstacle", "nominal"]:
        for metric in ["final_rauc", "rauc_aulc"]:
            rows.append(
                (
                    "Grid margin regime sensitivity 15-seed",
                    f"regime={regime}",
                    metric,
                    "bgr",
                    "uniform",
                    True,
                    15,
                    0,
                )
            )
            rows.append(
                (
                    "Grid margin regime sensitivity 30-seed",
                    f"regime={regime}",
                    metric,
                    "bgr",
                    "uniform",
                    True,
                    30,
                    0,
                )
            )
    for learning_rate, supports_final_rauc in [(0.015, True), (0.030, True), (0.060, False)]:
        rows.append(
            (
                "Grid margin learning-rate sensitivity 15-seed",
                f"learning_rate={learning_rate:.3f}",
                "final_rauc",
                "bgr",
                "uniform",
                supports_final_rauc,
                15 if supports_final_rauc else 1,
                0 if supports_final_rauc else 14,
            )
        )
        rows.append(
            (
                "Grid margin learning-rate sensitivity 15-seed",
                f"learning_rate={learning_rate:.3f}",
                "rauc_aulc",
                "bgr",
                "uniform",
                True,
                15,
                0,
            )
        )
        rows.append(
            (
                "Grid margin learning-rate sensitivity 30-seed",
                f"learning_rate={learning_rate:.3f}",
                "final_rauc",
                "bgr",
                "uniform",
                supports_final_rauc,
                30 if supports_final_rauc else 1,
                0 if supports_final_rauc else 29,
            )
        )
        rows.append(
            (
                "Grid margin learning-rate sensitivity 30-seed",
                f"learning_rate={learning_rate:.3f}",
                "rauc_aulc",
                "bgr",
                "uniform",
                True,
                30,
                0,
            )
        )
    for stress_case in ["diffuse_boundary", "low_feasibility", "sharp_low_margin"]:
        for metric in ["final_rauc", "rauc_aulc"]:
            rows.append(
                (
                    "Grid margin stress sensitivity 15-seed",
                    f"stress_case={stress_case}",
                    metric,
                    "bgr",
                    "uniform",
                    True,
                    15,
                    0,
                )
            )
            rows.append(
                (
                    "Grid margin stress sensitivity 30-seed",
                    f"stress_case={stress_case}",
                    metric,
                    "bgr",
                    "uniform",
                    True,
                    30,
                    0,
                )
            )
    for step in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]:
        rows.append(("Grid margin learning curve 15-seed", f"step={step}", "rauc", "bgr", "uniform", True, 15, 0))
    return rows


if __name__ == "__main__":
    unittest.main()
