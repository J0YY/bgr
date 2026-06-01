import unittest

from scripts.summarize_libero_openvla_recovery import summarize


class LiberoOpenVLARecoverySummaryTest(unittest.TestCase):
    def test_summarizes_state_conditioned_curves(self):
        rows = [
            _row("native", "identity", {}, True),
            _row("occ_20", "occlusion", {"fraction": 0.2}, True),
            _row("occ_50", "occlusion", {"fraction": 0.5}, False),
            _row("blur_25", "blur", {"radius": 2.5}, False),
        ]
        summary = summarize(rows, source_name="unit")
        by_family = {row["family"]: row for row in summary["families"]}
        self.assertIn("occlusion", by_family)
        self.assertIn("blur", by_family)
        self.assertGreater(by_family["occlusion"]["rauc_mean"], by_family["blur"]["rauc_mean"])
        self.assertGreater(by_family["occlusion"]["r80_mean"], 0.0)


def _row(name, perturbation_type, params, success):
    return {
        "episode_idx": 0,
        "init_state_idx": 0,
        "perturbation_name": name,
        "perturbation_params": params,
        "perturbation_type": perturbation_type,
        "steps": 10,
        "success": success,
        "suite": "object",
        "task_idx": 0,
        "task_name": "pick_up_object",
    }


if __name__ == "__main__":
    unittest.main()
