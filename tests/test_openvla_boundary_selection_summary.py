import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_openvla_boundary_selection import _aggregate, _summarize_dir


class OpenVLABoundarySelectionSummaryTest(unittest.TestCase):
    def test_boundary_metrics_from_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "top_k_candidates.json").write_text(
                json.dumps(
                    [
                        {"observed_cf_rate": 0.5, "predicted_cf_rate": 0.6, "perturbation_type": "blur"},
                        {"observed_cf_rate": 1.0, "predicted_cf_rate": 0.9, "perturbation_type": "shift"},
                    ]
                ),
                encoding="utf-8",
            )
            (path / "proposal_guided_summary.json").write_text(
                json.dumps({"n_counterfactual_certificates": 3}),
                encoding="utf-8",
            )
            row = _summarize_dir(
                path,
                method="bgr_boundary",
                lower=0.25,
                upper=0.75,
                summary_name="proposal_guided_summary.json",
            )
            self.assertEqual(row["method"], "bgr_boundary")
            self.assertEqual(row["boundary_hit_rate"], 0.5)
            self.assertEqual(row["mean_abs_distance_to_half"], 0.25)
            aggregate = _aggregate([row])
            self.assertEqual(aggregate[0]["runs"], 1)


if __name__ == "__main__":
    unittest.main()
