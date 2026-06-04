import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.merge_grid_margin_trials import main as merge_main


def result(method: str, seed: int, value: float) -> dict:
    return {
        "method": method,
        "seed": seed,
        "final_clean": value,
        "final_rauc": value + 0.1,
        "final_median_r80": value + 0.2,
        "rauc_aulc": value + 0.3,
        "best_rauc": value + 0.4,
        "history": [],
    }


def write_trial(root: Path, method: str, seed: int, value: float) -> None:
    trials = root / "out" / "trials"
    trials.mkdir(parents=True, exist_ok=True)
    (trials / f"{method}_seed{seed}.json").write_text(
        json.dumps({"method": method, "seed": seed, "result": result(method, seed, value)}),
        encoding="utf-8",
    )


class GridMarginTrialMergeTest(unittest.TestCase):
    def test_merge_requires_all_configured_trials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.yaml"
            config.write_text(
                "experiment:\n  methods: [uniform, bgr]\n  seeds: [0, 1]\n",
                encoding="utf-8",
            )
            write_trial(root, "uniform", 0, 0.1)

            with mock.patch(
                "sys.argv",
                [
                    "merge_grid_margin_trials.py",
                    "--config",
                    str(config),
                    "--out",
                    str(root / "out"),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "missing grid-margin trial"):
                    merge_main()

    def test_merge_writes_summary_and_results_in_config_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.yaml"
            config.write_text(
                "experiment:\n  methods: [uniform, bgr]\n  seeds: [0, 1]\n",
                encoding="utf-8",
            )
            write_trial(root, "bgr", 1, 0.4)
            write_trial(root, "uniform", 0, 0.1)
            write_trial(root, "bgr", 0, 0.3)
            write_trial(root, "uniform", 1, 0.2)

            with mock.patch(
                "sys.argv",
                [
                    "merge_grid_margin_trials.py",
                    "--config",
                    str(config),
                    "--out",
                    str(root / "out"),
                ],
            ):
                merge_main()

            summary = (root / "out" / "summary.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(summary[1].split(",")[:2], ["uniform", "0"])
            self.assertEqual(summary[2].split(",")[:2], ["uniform", "1"])
            self.assertEqual(summary[3].split(",")[:2], ["bgr", "0"])
            self.assertEqual(summary[4].split(",")[:2], ["bgr", "1"])
            payload = json.loads((root / "out" / "results.json").read_text(encoding="utf-8"))
            self.assertEqual([(row["method"], row["seed"]) for row in payload["results"]], [("uniform", 0), ("uniform", 1), ("bgr", 0), ("bgr", 1)])


if __name__ == "__main__":
    unittest.main()
