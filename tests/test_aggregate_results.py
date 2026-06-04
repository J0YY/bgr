import csv
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_results import (
    load_learning_rate_sensitivity,
    load_regime_sensitivity,
    load_stress_sensitivity,
    write_learning_rate_sensitivity_table,
    write_regime_sensitivity_table,
    write_stress_sensitivity_table,
)


class AggregateResultsTest(unittest.TestCase):
    def test_load_learning_rate_sensitivity_summarizes_methods_by_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.csv"
            write_csv(
                path,
                [
                    row("0.015", "uniform", "0", "0.90", "0.30", "0.20", "0.25"),
                    row("0.015", "uniform", "1", "0.92", "0.32", "0.22", "0.27"),
                    row("0.015", "bgr", "0", "0.95", "0.40", "0.30", "0.35"),
                    row("0.015", "bgr", "1", "0.97", "0.42", "0.32", "0.37"),
                    row("0.060", "uniform", "0", "0.91", "0.50", "0.42", "0.38"),
                    row("0.060", "bgr", "0", "0.94", "0.48", "0.40", "0.39"),
                ],
            )

            rows = load_learning_rate_sensitivity(path)
            table_path = Path(temp_dir) / "table.tex"
            write_learning_rate_sensitivity_table(table_path, rows)
            table = table_path.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 4)
        first = rows[0]
        self.assertEqual(first["learning_rate"], 0.015)
        self.assertEqual(first["method"], "BGR")
        self.assertEqual(first["n"], 2)
        self.assertAlmostEqual(float(first["rauc_mean"]), 0.41)
        self.assertIn("LR & Method & Clean & RAUC", table)
        self.assertIn("0.060 & BGR", table)

    def test_load_regime_sensitivity_summarizes_methods_by_regime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.csv"
            write_csv(
                path,
                [
                    regime_row("low_obstacle", "uniform", "0", "0.90", "0.30", "0.20", "0.25"),
                    regime_row("low_obstacle", "uniform", "1", "0.92", "0.32", "0.22", "0.27"),
                    regime_row("low_obstacle", "bgr", "0", "0.95", "0.40", "0.30", "0.35"),
                    regime_row("low_obstacle", "bgr", "1", "0.97", "0.42", "0.32", "0.37"),
                    regime_row("high_obstacle", "uniform", "0", "0.91", "0.50", "0.42", "0.38"),
                    regime_row("high_obstacle", "bgr", "0", "0.94", "0.48", "0.40", "0.39"),
                ],
            )

            rows = load_regime_sensitivity(path)
            table_path = Path(temp_dir) / "table.tex"
            write_regime_sensitivity_table(table_path, rows)
            table = table_path.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 4)
        by_regime_method = {(row["regime"], row["method"]): row for row in rows}
        self.assertAlmostEqual(float(by_regime_method[("low_obstacle", "BGR")]["rauc_mean"]), 0.41)
        self.assertEqual(by_regime_method[("low_obstacle", "BGR")]["n"], 2)
        self.assertIn("Regime & Method & Clean & RAUC", table)
        self.assertIn("Low obstacle & BGR", table)

    def test_load_stress_sensitivity_summarizes_methods_by_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.csv"
            write_csv(
                path,
                [
                    stress_row("sharp_low_margin", "uniform", "0", "0.90", "0.30", "0.20", "0.25"),
                    stress_row("sharp_low_margin", "uniform", "1", "0.92", "0.32", "0.22", "0.27"),
                    stress_row("sharp_low_margin", "bgr", "0", "0.95", "0.40", "0.30", "0.35"),
                    stress_row("sharp_low_margin", "bgr", "1", "0.97", "0.42", "0.32", "0.37"),
                    stress_row("low_feasibility", "uniform", "0", "0.91", "0.50", "0.42", "0.38"),
                    stress_row("low_feasibility", "bgr", "0", "0.94", "0.48", "0.40", "0.39"),
                ],
            )

            rows = load_stress_sensitivity(path)
            table_path = Path(temp_dir) / "table.tex"
            write_stress_sensitivity_table(table_path, rows)
            table = table_path.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 4)
        by_case_method = {(row["stress_case"], row["method"]): row for row in rows}
        self.assertAlmostEqual(float(by_case_method[("sharp_low_margin", "BGR")]["rauc_mean"]), 0.41)
        self.assertEqual(by_case_method[("sharp_low_margin", "BGR")]["n"], 2)
        self.assertIn("Stress case & Method & Clean & RAUC", table)
        self.assertIn("Sharp low-margin & BGR", table)


def row(
    learning_rate: str,
    method: str,
    seed: str,
    final_clean: str,
    final_rauc: str,
    final_median_r80: str,
    rauc_aulc: str,
) -> dict[str, str]:
    return {
        "learning_rate": learning_rate,
        "method": method,
        "seed": seed,
        "final_clean": final_clean,
        "final_rauc": final_rauc,
        "final_median_r80": final_median_r80,
        "rauc_aulc": rauc_aulc,
        "best_rauc": final_rauc,
    }


def regime_row(
    regime: str,
    method: str,
    seed: str,
    final_clean: str,
    final_rauc: str,
    final_median_r80: str,
    rauc_aulc: str,
) -> dict[str, str]:
    data = row("0.030", method, seed, final_clean, final_rauc, final_median_r80, rauc_aulc)
    data.pop("learning_rate")
    data.update({"regime": regime, "obstacle_prob": "0.22", "grid_size": "11", "max_offset": "6"})
    return data


def stress_row(
    stress_case: str,
    method: str,
    seed: str,
    final_clean: str,
    final_rauc: str,
    final_median_r80: str,
    rauc_aulc: str,
) -> dict[str, str]:
    data = row("0.030", method, seed, final_clean, final_rauc, final_median_r80, rauc_aulc)
    data.pop("learning_rate")
    data.update({"stress_case": stress_case})
    return data


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
