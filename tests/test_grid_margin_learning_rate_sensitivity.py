import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_grid_margin_learning_rate_sensitivity import config_for_learning_rate, main


class GridMarginLearningRateSensitivityTest(unittest.TestCase):
    def test_config_for_learning_rate_overrides_only_sweep_fields(self):
        base_config = {
            "experiment": {
                "name": "base",
                "seeds": [0, 1],
                "methods": ["uniform", "bgr"],
                "learning_rate": 0.03,
                "iterations": 10,
            },
            "bgr": {"radius_noise": 0.07},
        }

        config = config_for_learning_rate(base_config, learning_rate=0.06, seed=3, method="bgr")

        self.assertEqual(config["experiment"]["name"], "base_lr0.06")
        self.assertEqual(config["experiment"]["learning_rate"], 0.06)
        self.assertEqual(config["experiment"]["seeds"], [3])
        self.assertEqual(config["experiment"]["methods"], ["bgr"])
        self.assertEqual(base_config["experiment"]["learning_rate"], 0.03)
        self.assertEqual(base_config["experiment"]["seeds"], [0, 1])

    def test_main_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.yaml"
            sweep_path = root / "sweep.yaml"
            out_dir = root / "out"
            base_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  name: smoke",
                        "  seeds: [0]",
                        "  methods: [uniform, bgr]",
                        "  num_tasks: 4",
                        "  grid_size: 7",
                        "  obstacle_prob: 0.1",
                        "  replay_states_per_task: 2",
                        "  max_offset: 3",
                        "  iterations: 1",
                        "  eval_every: 1",
                        "  train_batch_size: 2",
                        "  eval_grid_size: 5",
                        "  alpha: 0.8",
                        "  learning_rate: 0.02",
                        "  target_margin: 0.38",
                        "bgr:",
                        "  initial_probes: [0.0, 0.5, 1.0]",
                        "  refresh_per_eval: 2",
                        "  min_trials: 1",
                        "  radius_noise: 0.05",
                    ]
                ),
                encoding="utf-8",
            )
            sweep_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        f"  base_config: {base_path}",
                        "  seeds: [0]",
                        "  methods: [uniform, bgr]",
                        "  learning_rates: [0.01, 0.02]",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "run_grid_margin_learning_rate_sensitivity.py",
                    "--config",
                    str(sweep_path),
                    "--out",
                    str(out_dir),
                ],
            ):
                main()

            rows = (out_dir / "summary.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 5)


if __name__ == "__main__":
    unittest.main()
