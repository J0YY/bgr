import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bgr.records import LevelRecord  # noqa: E402
from tools.reacher_recovery_probe import (  # noqa: E402
    LinearReacherPolicy,
    MAX_BIAS_NORM,
    MAX_WEIGHT_NORM,
    aggregate_rows,
    parse_ints,
    parse_strings,
    sample_pair,
)


class ReacherRecoveryProbeTest(unittest.TestCase):
    def test_parse_ints_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_ints("")

    def test_parse_strings_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_strings(" , ")

    def test_linear_policy_update_reduces_teacher_loss(self) -> None:
        policy = LinearReacherPolicy(kp=1.0, kd=0.1, torque_limit=0.65, learning_rate=0.05)
        features = np.array([0.5, -0.25, 0.1, -0.2], dtype=float)
        teacher = np.array([0.9, -0.7], dtype=float)

        before = np.mean((teacher - policy.predict_unclipped(features)) ** 2)
        policy.update(features, teacher)
        after = np.mean((teacher - policy.predict_unclipped(features)) ** 2)

        self.assertLess(after, before)

    def test_linear_policy_update_bounds_large_error(self) -> None:
        policy = LinearReacherPolicy(kp=1.0, kd=0.1, torque_limit=0.65, learning_rate=1.0)
        features = np.array([100.0, -100.0, 50.0, -50.0], dtype=float)
        teacher = np.array([1_000.0, -1_000.0], dtype=float)

        loss = policy.update(features, teacher)

        self.assertTrue(np.isfinite(loss))
        self.assertLessEqual(float(np.linalg.norm(policy.weights)), MAX_WEIGHT_NORM + 1e-9)
        self.assertLessEqual(float(np.max(np.abs(policy.bias))), MAX_BIAS_NORM + 1e-9)

    def test_aggregate_rows_reports_method_means(self) -> None:
        rows = [
            {
                "method": "bgr",
                "final_clean": 1.0,
                "final_rauc": 0.7,
                "final_median_r80": 2.0,
                "rauc_aulc": 0.6,
                "best_rauc": 0.8,
            },
            {
                "method": "bgr",
                "final_clean": 0.5,
                "final_rauc": 0.9,
                "final_median_r80": 4.0,
                "rauc_aulc": 0.8,
                "best_rauc": 0.9,
            },
        ]

        aggregate = aggregate_rows(rows)

        self.assertEqual(aggregate[0]["method"], "bgr")
        self.assertEqual(aggregate[0]["seeds"], "2")
        self.assertEqual(aggregate[0]["final_rauc_mean"], "0.800000")
        self.assertEqual(aggregate[0]["final_median_r80_mean"], "3.000000")

    def test_sample_pair_bgr_uniform_radius_uses_state_priority_and_uniform_radius(self) -> None:
        class FakeBench:
            pass

        records = [
            LevelRecord(
                id="low",
                domain="test",
                task_id="reacher",
                clean_success_hat=1.0,
                feasibility_hat=1.0,
                r_alpha_hat=0.0,
                sharpness_hat=0.1,
                uncertainty_hat=0.1,
            ),
            LevelRecord(
                id="target",
                domain="test",
                task_id="reacher",
                clean_success_hat=1.0,
                feasibility_hat=1.0,
                r_alpha_hat=3.0,
                sharpness_hat=1.0,
                uncertainty_hat=1.0,
            ),
        ]
        args = Namespace(
            max_radius=4.0,
            priority_temperature=0.5,
            uniform_mix=0.0,
            target_radius=3.0,
            radius_bandwidth=0.5,
            radius_noise=0.1,
            radius_uniform_mix=0.0,
        )
        scorer = type(
            "Scorer",
            (),
            {"score": lambda _self, record, _step: 100.0 if record.id == "target" else 1e-6},
        )()

        replay_idx, sigma = sample_pair("bgr_uniform_radius", FakeBench(), records, scorer, np.random.default_rng(7), args, 0)

        self.assertEqual(replay_idx, 1)
        self.assertGreaterEqual(sigma, 0.0)
        self.assertLessEqual(sigma, 4.0)


if __name__ == "__main__":
    unittest.main()
