import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_significance import (
    COMPARISONS,
    analyze,
    analyze_grid_learning_curve,
    analyze_learning_rate_sensitivity,
    analyze_learning_rate_sensitivity_30,
    analyze_regime_sensitivity,
    analyze_regime_sensitivity_30,
    analyze_stress_sensitivity,
    analyze_stress_sensitivity_30,
    analyze_target_sensitivity,
    analyze_target_sensitivity_30,
    format_pvalue_latex,
    result_row,
    write_latex,
    win_loss_tie_counts,
)


class AnalyzeSignificanceTest(unittest.TestCase):
    def test_result_row_includes_paired_confidence_interval(self):
        row = result_row("bench", "", "metric", "bgr", "uniform", [0.1, 0.2, 0.3], "higher")

        self.assertEqual(row["mean_treatment_minus_baseline"], "0.200000")
        self.assertIn("paired_ci95_low", row)
        self.assertIn("paired_ci95_high", row)
        self.assertEqual(row["paired_wins"], "3")
        self.assertEqual(row["paired_losses"], "0")
        self.assertEqual(row["paired_ties"], "0")
        self.assertLess(float(row["paired_ci95_low"]), 0.2)
        self.assertGreater(float(row["paired_ci95_high"]), 0.2)

    def test_win_loss_tie_counts_respect_direction(self):
        self.assertEqual(win_loss_tie_counts([0.1, -0.2, 0.0], "higher"), (1, 1, 1))
        self.assertEqual(win_loss_tie_counts([0.1, -0.2, 0.0], "lower"), (1, 1, 1))

    def test_latex_pvalue_format_avoids_rounded_zero(self):
        self.assertEqual(format_pvalue_latex("1.86265e-09"), "$<0.0001$")
        self.assertEqual(format_pvalue_latex("0.001"), "0.0010")

    def test_latex_table_includes_suffix_full_baselines_when_available(self):
        rows = [
            result_row("Synthetic margin 15-seed", "", "final_rauc", "bgr", "uniform", [0.1, 0.2], "higher"),
            result_row("Grid margin 15-seed", "", "final_rauc", "bgr", "uniform", [0.1, 0.2], "higher"),
            result_row("Estimator 15-seed", "", "boundary_hit_rate", "active", "uniform", [0.1, 0.2], "higher"),
        ]
        for baseline in ["clean_ft", "fixed", "failure_only", "loss_priority", "uniform"]:
            rows.append(
                result_row(
                    "Robot suffix coverage-full 30-seed",
                    "",
                    "final_rauc",
                    "bgr_broad",
                    baseline,
                    [0.1, 0.2],
                    "higher",
                )
            )
        rows.append(
            result_row(
                "Robot suffix coverage-full 30-seed",
                "",
                "final_transfer_rauc",
                "bgr_broad",
                "uniform",
                [0.1, 0.2],
                "higher",
            )
        )
        rows.append(
            result_row(
                "Robot suffix coverage-full 30-seed",
                "",
                "rauc_aulc",
                "bgr_broad",
                "uniform",
                [0.1, 0.2],
                "higher",
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "significance_table.tex"
            write_latex(rows, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("Comparison & Metric", text)
        self.assertIn("Suffix RAUC vs clean-only & final RAUC", text)
        self.assertIn("Suffix RAUC vs loss-priority & final RAUC", text)
        self.assertIn("Suffix transfer vs uniform & transfer RAUC", text)

    def test_suffix_coverage_full_comparisons_are_declared(self):
        comparisons = {
            (comparison.treatment, comparison.baseline, comparison.metric)
            for comparison in COMPARISONS
            if comparison.benchmark == "Robot suffix coverage-full 30-seed"
        }

        self.assertIn(("bgr_broad", "clean_ft", "final_rauc"), comparisons)
        self.assertIn(("bgr_broad", "uniform", "final_rauc"), comparisons)
        self.assertIn(("bgr_broad", "uniform", "final_transfer_rauc"), comparisons)
        self.assertIn(("bgr_broad", "uniform", "rauc_aulc"), comparisons)

    def test_suffix_coverage_full_replication_comparisons_are_declared(self):
        comparisons = {
            (comparison.treatment, comparison.baseline, comparison.metric)
            for comparison in COMPARISONS
            if comparison.benchmark == "Robot suffix coverage-full replication 30-seed"
        }

        self.assertIn(("bgr_broad", "clean_ft", "final_rauc"), comparisons)
        self.assertIn(("bgr_broad", "fixed", "final_rauc"), comparisons)
        self.assertIn(("bgr_broad", "failure_only", "final_rauc"), comparisons)
        self.assertIn(("bgr_broad", "loss_priority", "final_rauc"), comparisons)
        self.assertIn(("bgr_broad", "uniform", "final_rauc"), comparisons)
        self.assertIn(("bgr_broad", "uniform", "final_transfer_rauc"), comparisons)
        self.assertIn(("bgr_broad", "uniform", "rauc_aulc"), comparisons)

    def test_grid_margin_full_30_comparisons_are_declared(self):
        comparisons = {
            (comparison.treatment, comparison.baseline, comparison.metric)
            for comparison in COMPARISONS
            if comparison.benchmark == "Grid margin full 30-seed"
        }

        self.assertIn(("bgr", "uniform", "final_rauc"), comparisons)
        self.assertIn(("bgr", "uniform", "rauc_aulc"), comparisons)
        self.assertIn(("bgr", "fixed", "final_rauc"), comparisons)
        self.assertIn(("bgr", "failure_only", "final_rauc"), comparisons)
        self.assertIn(("bgr", "plr_loss", "final_rauc"), comparisons)

    def test_synthetic_and_estimator_30_confirmations_are_declared(self):
        synthetic = {
            (comparison.treatment, comparison.baseline, comparison.metric)
            for comparison in COMPARISONS
            if comparison.benchmark == "Synthetic margin 30-seed"
        }
        estimator = {
            (comparison.treatment, comparison.baseline, comparison.metric, comparison.direction)
            for comparison in COMPARISONS
            if comparison.benchmark == "Estimator 30-seed"
        }

        self.assertEqual(
            synthetic,
            {
                ("bgr", "uniform", "final_rauc"),
                ("bgr", "uniform", "rauc_aulc"),
                ("bgr", "uniform", "final_clean"),
                ("bgr", "fixed", "final_rauc"),
                ("bgr", "failure_only", "final_rauc"),
                ("bgr", "plr_loss", "final_rauc"),
            },
        )
        self.assertEqual(
            estimator,
            {
                ("active", "uniform", "boundary_hit_rate", "higher"),
                ("active", "uniform", "r80_mae", "lower"),
                ("active", "uniform", "rauc_mae", "lower"),
            },
        )

    def test_held_out_replication_comparisons_are_declared(self):
        grid = {
            (comparison.treatment, comparison.baseline, comparison.metric)
            for comparison in COMPARISONS
            if comparison.benchmark == "Grid margin replication 30-seed"
        }
        suffix = {
            (comparison.treatment, comparison.baseline, comparison.metric)
            for comparison in COMPARISONS
            if comparison.benchmark == "Robot suffix replication 30-seed"
        }

        self.assertEqual(
            grid,
            {
                ("bgr", "uniform", "final_rauc"),
                ("bgr", "uniform", "rauc_aulc"),
                ("bgr", "uniform", "final_clean"),
            },
        )
        self.assertEqual(
            suffix,
            {
                ("bgr_broad", "uniform", "final_clean"),
                ("bgr_broad", "uniform", "final_rauc"),
                ("bgr_broad", "uniform", "final_transfer_rauc"),
                ("bgr_broad", "uniform", "rauc_aulc"),
            },
        )

    def test_analyze_includes_completed_full_baseline_confirmations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_required_comparison_summaries(root)

            rows = analyze(root)

        self.assertTrue(any(row["benchmark"] == "Robot suffix coverage-full 30-seed" for row in rows))
        self.assertTrue(any(row["benchmark"] == "Robot suffix coverage-full replication 30-seed" for row in rows))
        self.assertTrue(any(row["benchmark"] == "Grid margin full 30-seed" for row in rows))
        self.assertTrue(any(row["benchmark"] == "Robot suffix replication 30-seed" for row in rows))
        self.assertTrue(any(row["benchmark"] == "Grid margin replication 30-seed" for row in rows))
        self.assertTrue(any(row["benchmark"] == "Synthetic margin 30-seed" for row in rows))
        self.assertTrue(any(row["benchmark"] == "Estimator 30-seed" for row in rows))

    def test_analyze_includes_suffix_coverage_full_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_required_comparison_summaries(root)

            rows = analyze(root)

        suffix_full = [row for row in rows if row["benchmark"] == "Robot suffix coverage-full 30-seed"]
        expected_count = sum(
            1
            for comparison in COMPARISONS
            if comparison.benchmark == "Robot suffix coverage-full 30-seed"
        )
        self.assertEqual(len(suffix_full), expected_count)
        by_baseline_metric = {(row["baseline"], row["metric"]): row for row in suffix_full}
        self.assertEqual(by_baseline_metric[("uniform", "final_rauc")]["n"], "2")
        self.assertEqual(by_baseline_metric[("uniform", "final_rauc")]["mean_treatment_minus_baseline"], "0.100000")

    def test_target_sensitivity_compares_each_target_to_uniform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sensitivity_dir = root / "grid_margin_target_sensitivity_15seed_v1"
            baseline_dir = root / "grid_margin_full_15seed_v1"
            sensitivity_dir.mkdir()
            baseline_dir.mkdir()

            write_csv(
                sensitivity_dir / "summary.csv",
                [
                    {
                        "method": "bgr",
                        "target_margin": "0.26",
                        "seed": "0",
                        "final_rauc": "0.44",
                        "rauc_aulc": "0.36",
                    },
                    {
                        "method": "bgr",
                        "target_margin": "0.26",
                        "seed": "1",
                        "final_rauc": "0.45",
                        "rauc_aulc": "0.37",
                    },
                    {
                        "method": "bgr",
                        "target_margin": "0.38",
                        "seed": "0",
                        "final_rauc": "0.43",
                        "rauc_aulc": "0.35",
                    },
                    {
                        "method": "bgr",
                        "target_margin": "0.38",
                        "seed": "1",
                        "final_rauc": "0.44",
                        "rauc_aulc": "0.36",
                    },
                ],
            )
            write_csv(
                baseline_dir / "summary.csv",
                [
                    {
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.40",
                        "rauc_aulc": "0.31",
                    },
                    {
                        "method": "uniform",
                        "seed": "1",
                        "final_rauc": "0.41",
                        "rauc_aulc": "0.32",
                    },
                ],
            )

            rows = analyze_target_sensitivity(root)

        self.assertEqual(len(rows), 4)
        conditions = {(row["condition"], row["metric"]) for row in rows}
        self.assertEqual(
            conditions,
            {
                ("target_margin=0.26", "final_rauc"),
                ("target_margin=0.26", "rauc_aulc"),
                ("target_margin=0.38", "final_rauc"),
                ("target_margin=0.38", "rauc_aulc"),
            },
        )
        self.assertTrue(all(row["supports_treatment"] == "true" for row in rows))
        self.assertTrue(all(row["baseline"] == "uniform" for row in rows))

    def test_target_sensitivity_30_compares_clean_rauc_and_aulc_to_30seed_uniform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sensitivity_dir = root / "grid_margin_target_sensitivity_30seed_v1"
            baseline_dir = root / "grid_margin_full_30seed_v1"
            sensitivity_dir.mkdir()
            baseline_dir.mkdir()

            write_csv(
                sensitivity_dir / "summary.csv",
                [
                    {
                        "method": "bgr",
                        "target_margin": "0.26",
                        "seed": "0",
                        "final_clean": "0.95",
                        "final_rauc": "0.44",
                        "rauc_aulc": "0.36",
                    },
                    {
                        "method": "bgr",
                        "target_margin": "0.26",
                        "seed": "1",
                        "final_clean": "0.96",
                        "final_rauc": "0.45",
                        "rauc_aulc": "0.37",
                    },
                    {
                        "method": "bgr",
                        "target_margin": "0.54",
                        "seed": "0",
                        "final_clean": "0.93",
                        "final_rauc": "0.42",
                        "rauc_aulc": "0.34",
                    },
                    {
                        "method": "bgr",
                        "target_margin": "0.54",
                        "seed": "1",
                        "final_clean": "0.94",
                        "final_rauc": "0.43",
                        "rauc_aulc": "0.35",
                    },
                ],
            )
            write_csv(
                baseline_dir / "summary.csv",
                [
                    {
                        "method": "uniform",
                        "seed": "0",
                        "final_clean": "0.90",
                        "final_rauc": "0.40",
                        "rauc_aulc": "0.31",
                    },
                    {
                        "method": "uniform",
                        "seed": "1",
                        "final_clean": "0.91",
                        "final_rauc": "0.41",
                        "rauc_aulc": "0.32",
                    },
                ],
            )

            rows = analyze_target_sensitivity_30(root)

        self.assertEqual(len(rows), 6)
        conditions = {(row["condition"], row["metric"]) for row in rows}
        self.assertEqual(
            conditions,
            {
                ("target_margin=0.26", "final_clean"),
                ("target_margin=0.26", "final_rauc"),
                ("target_margin=0.26", "rauc_aulc"),
                ("target_margin=0.54", "final_clean"),
                ("target_margin=0.54", "final_rauc"),
                ("target_margin=0.54", "rauc_aulc"),
            },
        )
        self.assertTrue(all(row["baseline"] == "uniform" for row in rows))
        self.assertTrue(all(row["benchmark"] == "Grid margin target sensitivity 30-seed" for row in rows))

    def test_grid_learning_curve_compares_each_eval_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "grid_margin_full_15seed_v1"
            run_dir.mkdir()
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "method": "bgr",
                                "seed": 0,
                                "history": [
                                    {"step": 0.0, "rauc": 0.20},
                                    {"step": 30.0, "rauc": 0.30},
                                    {"step": 60.0, "rauc": 0.35},
                                ],
                            },
                            {
                                "method": "uniform",
                                "seed": 0,
                                "history": [
                                    {"step": 0.0, "rauc": 0.20},
                                    {"step": 30.0, "rauc": 0.28},
                                    {"step": 60.0, "rauc": 0.31},
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = analyze_grid_learning_curve(root)

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["condition"] for row in rows], ["step=30", "step=60"])
        self.assertEqual([row["mean_treatment_minus_baseline"] for row in rows], ["0.020000", "0.040000"])
        self.assertTrue(all(row["supports_treatment"] == "true" for row in rows))

    def test_learning_rate_sensitivity_compares_each_rate_and_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "grid_margin_learning_rate_sensitivity_15seed_v1"
            run_dir.mkdir()
            write_csv(
                run_dir / "summary.csv",
                [
                    {
                        "learning_rate": "0.015",
                        "method": "bgr",
                        "seed": "0",
                        "final_clean": "0.95",
                        "final_rauc": "0.42",
                        "final_median_r80": "0.34",
                        "rauc_aulc": "0.33",
                    },
                    {
                        "learning_rate": "0.015",
                        "method": "uniform",
                        "seed": "0",
                        "final_clean": "0.90",
                        "final_rauc": "0.38",
                        "final_median_r80": "0.30",
                        "rauc_aulc": "0.30",
                    },
                    {
                        "learning_rate": "0.060",
                        "method": "bgr",
                        "seed": "0",
                        "final_clean": "0.94",
                        "final_rauc": "0.48",
                        "final_median_r80": "0.41",
                        "rauc_aulc": "0.39",
                    },
                    {
                        "learning_rate": "0.060",
                        "method": "uniform",
                        "seed": "0",
                        "final_clean": "0.90",
                        "final_rauc": "0.49",
                        "final_median_r80": "0.43",
                        "rauc_aulc": "0.38",
                    },
                ],
            )

            rows = analyze_learning_rate_sensitivity(root)

        self.assertEqual(len(rows), 8)
        by_condition_metric = {(row["condition"], row["metric"]): row for row in rows}
        self.assertEqual(
            by_condition_metric[("learning_rate=0.015", "final_rauc")]["mean_treatment_minus_baseline"],
            "0.040000",
        )
        self.assertEqual(
            by_condition_metric[("learning_rate=0.060", "final_rauc")]["supports_treatment"],
            "false",
        )
        self.assertEqual(
            by_condition_metric[("learning_rate=0.060", "rauc_aulc")]["supports_treatment"],
            "true",
        )

    def test_learning_rate_sensitivity_30_compares_each_rate_and_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "grid_margin_learning_rate_sensitivity_30seed_v1"
            run_dir.mkdir()
            write_csv(
                run_dir / "summary.csv",
                [
                    {
                        "learning_rate": "0.015",
                        "method": "bgr",
                        "seed": "0",
                        "final_clean": "0.95",
                        "final_rauc": "0.42",
                        "final_median_r80": "0.34",
                        "rauc_aulc": "0.33",
                    },
                    {
                        "learning_rate": "0.015",
                        "method": "uniform",
                        "seed": "0",
                        "final_clean": "0.90",
                        "final_rauc": "0.38",
                        "final_median_r80": "0.30",
                        "rauc_aulc": "0.30",
                    },
                    {
                        "learning_rate": "0.060",
                        "method": "bgr",
                        "seed": "0",
                        "final_clean": "0.94",
                        "final_rauc": "0.48",
                        "final_median_r80": "0.41",
                        "rauc_aulc": "0.39",
                    },
                    {
                        "learning_rate": "0.060",
                        "method": "uniform",
                        "seed": "0",
                        "final_clean": "0.90",
                        "final_rauc": "0.49",
                        "final_median_r80": "0.43",
                        "rauc_aulc": "0.38",
                    },
                ],
            )

            rows = analyze_learning_rate_sensitivity_30(root)

        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["benchmark"] == "Grid margin learning-rate sensitivity 30-seed" for row in rows))
        by_condition_metric = {(row["condition"], row["metric"]): row for row in rows}
        self.assertEqual(
            by_condition_metric[("learning_rate=0.015", "final_rauc")]["mean_treatment_minus_baseline"],
            "0.040000",
        )
        self.assertEqual(
            by_condition_metric[("learning_rate=0.060", "final_rauc")]["supports_treatment"],
            "false",
        )
        self.assertEqual(
            by_condition_metric[("learning_rate=0.060", "rauc_aulc")]["supports_treatment"],
            "true",
        )

    def test_regime_sensitivity_compares_each_regime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "grid_margin_regime_sensitivity_15seed_v1"
            run_dir.mkdir()
            write_csv(
                run_dir / "summary.csv",
                [
                    {
                        "regime": "low_obstacle",
                        "method": "bgr",
                        "seed": "0",
                        "final_rauc": "0.42",
                        "rauc_aulc": "0.33",
                    },
                    {
                        "regime": "low_obstacle",
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.38",
                        "rauc_aulc": "0.30",
                    },
                    {
                        "regime": "high_obstacle",
                        "method": "bgr",
                        "seed": "0",
                        "final_rauc": "0.48",
                        "rauc_aulc": "0.39",
                    },
                    {
                        "regime": "high_obstacle",
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.49",
                        "rauc_aulc": "0.38",
                    },
                ],
            )

            rows = analyze_regime_sensitivity(root)

        self.assertEqual(len(rows), 4)
        by_condition_metric = {(row["condition"], row["metric"]): row for row in rows}
        self.assertEqual(
            by_condition_metric[("regime=low_obstacle", "final_rauc")]["mean_treatment_minus_baseline"],
            "0.040000",
        )
        self.assertEqual(
            by_condition_metric[("regime=high_obstacle", "final_rauc")]["supports_treatment"],
            "false",
        )
        self.assertEqual(
            by_condition_metric[("regime=high_obstacle", "rauc_aulc")]["supports_treatment"],
            "true",
        )

    def test_regime_sensitivity_30_compares_each_regime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "grid_margin_regime_sensitivity_30seed_v1"
            run_dir.mkdir()
            write_csv(
                run_dir / "summary.csv",
                [
                    {
                        "regime": "low_obstacle",
                        "method": "bgr",
                        "seed": "0",
                        "final_rauc": "0.42",
                        "rauc_aulc": "0.33",
                    },
                    {
                        "regime": "low_obstacle",
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.38",
                        "rauc_aulc": "0.30",
                    },
                    {
                        "regime": "high_obstacle",
                        "method": "bgr",
                        "seed": "0",
                        "final_rauc": "0.48",
                        "rauc_aulc": "0.39",
                    },
                    {
                        "regime": "high_obstacle",
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.49",
                        "rauc_aulc": "0.38",
                    },
                ],
            )

            rows = analyze_regime_sensitivity_30(root)

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["benchmark"] == "Grid margin regime sensitivity 30-seed" for row in rows))
        by_condition_metric = {(row["condition"], row["metric"]): row for row in rows}
        self.assertEqual(
            by_condition_metric[("regime=low_obstacle", "final_rauc")]["mean_treatment_minus_baseline"],
            "0.040000",
        )
        self.assertEqual(
            by_condition_metric[("regime=high_obstacle", "final_rauc")]["supports_treatment"],
            "false",
        )
        self.assertEqual(
            by_condition_metric[("regime=high_obstacle", "rauc_aulc")]["supports_treatment"],
            "true",
        )

    def test_stress_sensitivity_compares_each_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "grid_margin_stress_sensitivity_15seed_v1"
            run_dir.mkdir()
            write_csv(
                run_dir / "summary.csv",
                [
                    {
                        "stress_case": "sharp_low_margin",
                        "method": "bgr",
                        "seed": "0",
                        "final_rauc": "0.42",
                        "rauc_aulc": "0.33",
                    },
                    {
                        "stress_case": "sharp_low_margin",
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.38",
                        "rauc_aulc": "0.30",
                    },
                    {
                        "stress_case": "low_feasibility",
                        "method": "bgr",
                        "seed": "0",
                        "final_rauc": "0.48",
                        "rauc_aulc": "0.39",
                    },
                    {
                        "stress_case": "low_feasibility",
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.49",
                        "rauc_aulc": "0.38",
                    },
                ],
            )

            rows = analyze_stress_sensitivity(root)

        self.assertEqual(len(rows), 4)
        by_condition_metric = {(row["condition"], row["metric"]): row for row in rows}
        self.assertEqual(
            by_condition_metric[("stress_case=sharp_low_margin", "final_rauc")]["mean_treatment_minus_baseline"],
            "0.040000",
        )
        self.assertEqual(
            by_condition_metric[("stress_case=low_feasibility", "final_rauc")]["supports_treatment"],
            "false",
        )
        self.assertEqual(
            by_condition_metric[("stress_case=low_feasibility", "rauc_aulc")]["supports_treatment"],
            "true",
        )

    def test_stress_sensitivity_30_compares_each_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "grid_margin_stress_sensitivity_30seed_v1"
            run_dir.mkdir()
            write_csv(
                run_dir / "summary.csv",
                [
                    {
                        "stress_case": "sharp_low_margin",
                        "method": "bgr",
                        "seed": "0",
                        "final_rauc": "0.42",
                        "rauc_aulc": "0.33",
                    },
                    {
                        "stress_case": "sharp_low_margin",
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.38",
                        "rauc_aulc": "0.30",
                    },
                    {
                        "stress_case": "low_feasibility",
                        "method": "bgr",
                        "seed": "0",
                        "final_rauc": "0.48",
                        "rauc_aulc": "0.39",
                    },
                    {
                        "stress_case": "low_feasibility",
                        "method": "uniform",
                        "seed": "0",
                        "final_rauc": "0.49",
                        "rauc_aulc": "0.38",
                    },
                ],
            )

            rows = analyze_stress_sensitivity_30(root)

        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["benchmark"] == "Grid margin stress sensitivity 30-seed" for row in rows))
        by_condition_metric = {(row["condition"], row["metric"]): row for row in rows}
        self.assertEqual(
            by_condition_metric[("stress_case=sharp_low_margin", "final_rauc")]["mean_treatment_minus_baseline"],
            "0.040000",
        )
        self.assertEqual(
            by_condition_metric[("stress_case=low_feasibility", "final_rauc")]["supports_treatment"],
            "false",
        )
        self.assertEqual(
            by_condition_metric[("stress_case=low_feasibility", "rauc_aulc")]["supports_treatment"],
            "true",
        )


def write_required_comparison_summaries(root: Path) -> None:
    runs = {
        "toy_15seed_v1": ["bgr", "uniform", "fixed", "failure_only", "plr_loss"],
        "toy_30seed_v1": ["bgr", "uniform", "fixed", "failure_only", "plr_loss"],
        "grid_margin_pair_15seed_v1": ["bgr", "uniform"],
        "grid_margin_full_15seed_v1": ["bgr", "fixed", "failure_only", "plr_loss"],
        "grid_margin_full_30seed_v1": ["bgr", "uniform", "fixed", "failure_only", "plr_loss"],
        "grid_margin_full_replication_30seed_v1": ["bgr", "uniform"],
        "grid_margin_ablation_15seed_v1": ["bgr", "bgr_uniform_radius", "uniform", "bgr_no_uncertainty", "bgr_no_sharpness"],
        "suffix_full_15seed_v1": ["bgr", "clean_ft", "fixed", "failure_only", "loss_priority", "uniform"],
        "suffix_strategy_coverage_30seed_v1": ["bgr_broad", "uniform"],
        "suffix_strategy_coverage_replication_30seed_v1": ["bgr_broad", "uniform"],
        "suffix_coverage_full_30seed_v1": ["bgr_broad", "clean_ft", "uniform", "fixed", "failure_only", "loss_priority"],
        "suffix_coverage_full_replication_30seed_v1": ["bgr_broad", "clean_ft", "uniform", "fixed", "failure_only", "loss_priority"],
        "estimator_pair_15seed_v1": ["active", "uniform"],
        "estimator_pair_30seed_v1": ["active", "uniform"],
    }
    for run_name, methods in runs.items():
        run_dir = root / run_name
        run_dir.mkdir()
        rows = []
        for method in methods:
            for seed in range(2):
                rows.append(
                    {
                        "method": method,
                        "seed": str(seed),
                        "final_clean": "0.90" if method in {"bgr", "bgr_broad", "active"} else "0.80",
                        "final_rauc": "0.50" if method in {"bgr", "bgr_broad", "active"} else "0.40",
                        "final_median_r80": "0.30" if method in {"bgr", "bgr_broad", "active"} else "0.20",
                        "final_transfer_rauc": "0.32" if method in {"bgr", "bgr_broad", "active"} else "0.30",
                        "rauc_aulc": "0.38" if method in {"bgr", "bgr_broad", "active"} else "0.35",
                        "boundary_hit_rate": "0.70" if method == "active" else "0.60",
                        "r80_mae": "0.08" if method == "active" else "0.10",
                        "rauc_mae": "0.06" if method == "active" else "0.07",
                    }
                )
        write_csv(run_dir / "summary.csv", rows)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
