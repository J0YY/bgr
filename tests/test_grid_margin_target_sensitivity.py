import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts.run_grid_margin_target_sensitivity import config_for_target, main


class GridMarginTargetSensitivityTest(unittest.TestCase):
    def test_config_for_target_overrides_only_sweep_fields(self):
        base_config = {
            "experiment": {
                "seeds": [0, 1],
                "methods": ["uniform", "bgr"],
                "target_margin": 0.38,
                "iterations": 10,
            },
            "bgr": {"radius_noise": 0.07},
        }

        config = config_for_target(base_config, target_margin=0.26, seed=3, method="bgr")

        self.assertEqual(config["experiment"]["target_margin"], 0.26)
        self.assertEqual(config["experiment"]["seeds"], [3])
        self.assertEqual(config["experiment"]["methods"], ["bgr"])
        self.assertEqual(base_config["experiment"]["target_margin"], 0.38)
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
                        "  seeds: [0]",
                        "  methods: [bgr]",
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
                        "  method: bgr",
                        "  target_margins: [0.26, 0.38]",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "run_grid_margin_target_sensitivity.py",
                    "--config",
                    str(sweep_path),
                    "--out",
                    str(out_dir),
                ],
            ):
                main()

            self.assertTrue((out_dir / "summary.csv").exists())
            rows = (out_dir / "summary.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
