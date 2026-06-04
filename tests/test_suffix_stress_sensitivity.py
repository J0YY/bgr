import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_suffix_stress_sensitivity import (
    SuffixStressCase,
    config_for_stress_case,
    main,
    parse_stress_cases,
)


class SuffixStressSensitivityTest(unittest.TestCase):
    def test_parse_stress_cases_requires_named_overrides(self):
        with self.assertRaises(ValueError):
            parse_stress_cases({"experiment": {"stress_cases": [{"name": "empty"}]}})
        with self.assertRaises(ValueError):
            parse_stress_cases({"experiment": {"stress_cases": [{"temperature_scale": 1.4}]}})

    def test_config_for_stress_case_overrides_only_requested_fields(self):
        base_config = {
            "experiment": {
                "name": "base",
                "seeds": [0, 1],
                "methods": ["uniform", "bgr_broad"],
                "teacher_quality_min": 0.78,
                "teacher_quality_max": 0.98,
            },
            "bgr": {"radius_noise": 0.07},
        }
        case = SuffixStressCase("low_teacher", {"teacher_quality_min": 0.60, "teacher_quality_max": 0.82})

        config = config_for_stress_case(base_config, case, seed=3, method="bgr_broad")

        self.assertEqual(config["experiment"]["name"], "base_low_teacher")
        self.assertEqual(config["experiment"]["teacher_quality_min"], 0.60)
        self.assertEqual(config["experiment"]["teacher_quality_max"], 0.82)
        self.assertEqual(config["experiment"]["seeds"], [3])
        self.assertEqual(config["experiment"]["methods"], ["bgr_broad"])
        self.assertEqual(base_config["experiment"]["teacher_quality_min"], 0.78)

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
                        "  methods: [uniform, bgr_broad]",
                        "  num_tasks: 4",
                        "  suffixes_per_task: 2",
                        "  iterations: 1",
                        "  eval_every: 1",
                        "  train_batch_size: 2",
                        "  eval_grid_size: 5",
                        "  alpha: 0.8",
                        "  learning_rate: 0.02",
                        "  target_margin: 0.38",
                        "  baseline_candidates: 2",
                        "  failure_probe_trials: 1",
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
                        "  methods: [uniform, bgr_broad]",
                        '  stress_cases: [{"name": "low_teacher", "teacher_quality_min": 0.6, "teacher_quality_max": 0.82}, {"name": "diffuse", "temperature_scale": 1.4}]',
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "sys.argv",
                [
                    "run_suffix_stress_sensitivity.py",
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
