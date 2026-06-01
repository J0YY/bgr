import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_openvla_bgr_finetune_manifest import _load_candidates, _summarize


class OpenVLABGRFinetuneManifestTest(unittest.TestCase):
    def test_load_candidates_marks_boundary_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "top_k_candidates.json").write_text(
                json.dumps(
                    [
                        {
                            "name": "blur_mid",
                            "spec": "blur_mid:blur:1.0",
                            "perturbation_type": "blur",
                            "perturbation_params": {"radius": 1.0},
                            "observed_cf_rate": 0.5,
                            "predicted_cf_rate": 0.6,
                            "observed_cf_failures": 2,
                            "observed_episodes": 4,
                            "observed_success_rate": 0.5,
                        },
                        {
                            "name": "shift_easy",
                            "spec": "shift_easy:shift:0.01:0.0",
                            "perturbation_type": "shift",
                            "observed_cf_rate": 0.0,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            rows = _load_candidates(path, method="bgr_boundary", lower=0.25, upper=0.75)
            self.assertTrue(rows[0]["in_boundary_band"])
            self.assertEqual(rows[0]["training_role"], "boundary_candidate")
            self.assertEqual(rows[1]["training_role"], "coverage_candidate")
            summary = _summarize(rows)
            self.assertEqual(summary["bgr_boundary"]["boundary_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
