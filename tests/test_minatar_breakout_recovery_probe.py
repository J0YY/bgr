import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bgr.records import LevelRecord
from tools.minatar_breakout_recovery_probe import (
    LinearSoftmaxPolicy,
    aggregate_rows,
    parse_ints,
    parse_strings,
    sample_training_pair,
)


class MinAtarBreakoutRecoveryProbeTest(unittest.TestCase):
    def test_parse_ints_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_ints(" , ")

    def test_parse_strings_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_strings("")

    def test_linear_policy_update_reduces_teacher_loss(self) -> None:
        policy = LinearSoftmaxPolicy(rng=np.random.default_rng(0), learning_rate=0.20, init_noise=0.0, feature_dim=3)
        features = np.array([0.5, -0.25, 1.0], dtype=float)

        before = policy.loss(features, 2)
        policy.update(features, 2)
        after = policy.loss(features, 2)

        self.assertLess(after, before)

    def test_aggregate_rows_reports_method_means(self) -> None:
        rows = [
            {
                "method": "bgr",
                "final_clean": 1.0,
                "final_rauc": 0.5,
                "final_median_r80": 1.0,
                "rauc_aulc": 0.4,
                "best_rauc": 0.6,
            },
            {
                "method": "bgr",
                "final_clean": 0.5,
                "final_rauc": 0.7,
                "final_median_r80": 3.0,
                "rauc_aulc": 0.8,
                "best_rauc": 0.9,
            },
        ]

        aggregate = aggregate_rows(rows)

        self.assertEqual(aggregate[0]["method"], "bgr")
        self.assertEqual(aggregate[0]["seeds"], "2")
        self.assertEqual(aggregate[0]["final_rauc_mean"], "0.600000")
        self.assertEqual(aggregate[0]["final_median_r80_mean"], "2.000000")

    def test_sample_pair_bgr_uniform_radius_uses_state_priority_and_uniform_radius(self) -> None:
        class FakeBench:
            pass

        records = [
            LevelRecord(
                id="low",
                domain="test",
                task_id="minatar_breakout",
                clean_success_hat=1.0,
                feasibility_hat=1.0,
                r_alpha_hat=0.0,
                sharpness_hat=0.1,
                uncertainty_hat=0.1,
            ),
            LevelRecord(
                id="target",
                domain="test",
                task_id="minatar_breakout",
                clean_success_hat=1.0,
                feasibility_hat=1.0,
                r_alpha_hat=1.0,
                sharpness_hat=1.0,
                uncertainty_hat=1.0,
            ),
        ]
        args = Namespace(
            max_radius=5.0,
            priority_temperature=0.5,
            uniform_mix=0.0,
            radius_noise=0.35,
            radius_uniform_mix=0.0,
        )
        scorer = type(
            "Scorer",
            (),
            {"score": lambda _self, record, _step: 100.0 if record.id == "target" else 1e-6},
        )()

        replay_idx, sigma = sample_training_pair("bgr_uniform_radius", FakeBench(), records, scorer, np.random.default_rng(7), args, 0)

        self.assertEqual(replay_idx, 1)
        self.assertGreaterEqual(sigma, 0.0)
        self.assertLessEqual(sigma, 5.0)


if __name__ == "__main__":
    unittest.main()
