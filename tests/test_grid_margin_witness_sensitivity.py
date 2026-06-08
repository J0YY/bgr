import tempfile
import unittest
from pathlib import Path

from tools.grid_margin_witness_sensitivity import (
    Scenario,
    parse_scenarios,
    seeds_from_config,
    summarize,
    SensitivityRow,
    write_outputs,
)


class GridMarginWitnessSensitivityTest(unittest.TestCase):
    def test_parse_scenarios_validates_rates(self) -> None:
        with self.assertRaisesRegex(ValueError, "rates must be"):
            parse_scenarios("bad:1.2:0.0")

        scenarios = parse_scenarios("fp:0.1:0.0,sym:0.2:0.2")

        self.assertEqual(scenarios, [Scenario("fp", 0.1, 0.0), Scenario("sym", 0.2, 0.2)])

    def test_seeds_override_config(self) -> None:
        config = {"experiment": {"seeds": [0, 1, 2]}}

        self.assertEqual(seeds_from_config(config, "4,5"), [4, 5])
        self.assertEqual(seeds_from_config(config, None), [0, 1, 2])

    def test_summarize_reports_drop_vs_exact(self) -> None:
        rows = [
            SensitivityRow("exact", 0, 10, 10, 1.0, 1.0, 0.0, 0.8, 1.0, 0.05, 0.9, 0.4, 0.3),
            SensitivityRow("symmetric", 0, 10, 8, 0.8, 0.75, 0.25, 0.6, 0.5, 0.07, 0.9, 0.4, 0.3),
        ]

        summary = {row["scenario"]: row for row in summarize(rows)}

        self.assertEqual(summary["exact"]["valid_rate_drop_vs_exact"], "0.000000")
        self.assertEqual(summary["symmetric"]["valid_rate_drop_vs_exact"], "-0.250000")
        self.assertEqual(summary["symmetric"]["recall_drop_vs_exact"], "-0.500000")

    def test_write_outputs(self) -> None:
        rows = [SensitivityRow("exact", 0, 10, 10, 1.0, 1.0, 0.0, 0.8, 1.0, 0.05, 0.9, 0.4, 0.3)]
        summary = summarize(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(out, rows, summary, {"experiment": {"seeds": [0]}})

            self.assertTrue((out / "sample_rows.csv").exists())
            self.assertTrue((out / "summary.csv").exists())
            self.assertTrue((out / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
