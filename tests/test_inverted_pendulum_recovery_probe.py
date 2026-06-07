import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bgr.records import LevelRecord  # noqa: E402
from tools.inverted_pendulum_recovery_probe import (  # noqa: E402
    LinearBalancePolicy,
    MAX_BIAS_NORM,
    MAX_WEIGHT_NORM,
    aggregate_rows,
    parse_ints,
    parse_strings,
    sample_pair,
)


class InvertedPendulumRecoveryProbeTest(unittest.TestCase):
    def test_parse_ints_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_ints("")

    def test_parse_strings_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_strings(" , ")

    def test_linear_policy_update_reduces_teacher_loss(self) -> None:
        policy = LinearBalancePolicy(
            pole_kp=5.0,
            pole_kd=1.0,
            cart_kp=0.4,
            cart_kd=0.05,
            action_low=-3.0,
            action_high=3.0,
            learning_rate=0.05,
        )
        features = np.array([0.2, 0.08, 0.1, 0.3], dtype=float)
        teacher = 2.5

        before = (teacher - policy.predict_unclipped(features)) ** 2
        policy.update(features, teacher)
        after = (teacher - policy.predict_unclipped(features)) ** 2

        self.assertLess(after, before)

    def test_linear_policy_update_bounds_large_error(self) -> None:
        policy = LinearBalancePolicy(
            pole_kp=5.0,
            pole_kd=1.0,
            cart_kp=0.4,
            cart_kd=0.05,
            action_low=-3.0,
            action_high=3.0,
            learning_rate=1.0,
        )
        features = np.array([100.0, -100.0, 50.0, -50.0], dtype=float)

        loss = policy.update(features, 1_000.0)

        self.assertTrue(np.isfinite(loss))
        self.assertLessEqual(float(np.linalg.norm(policy.weights)), MAX_WEIGHT_NORM + 1e-9)
        self.assertLessEqual(abs(policy.bias), MAX_BIAS_NORM + 1e-9)

    def test_aggregate_rows_reports_method_means(self) -> None:
        rows = [
            {
                "method": "bgr",
                "final_clean": 1.0,
                "final_rauc": 0.3,
                "final_median_r80": 0.1,
                "rauc_aulc": 0.2,
                "best_rauc": 0.4,
            },
            {
                "method": "bgr",
                "final_clean": 0.5,
                "final_rauc": 0.5,
                "final_median_r80": 0.2,
                "rauc_aulc": 0.4,
                "best_rauc": 0.6,
            },
        ]

        aggregate = aggregate_rows(rows)

        self.assertEqual(aggregate[0]["method"], "bgr")
        self.assertEqual(aggregate[0]["seeds"], "2")
        self.assertEqual(aggregate[0]["final_rauc_mean"], "0.400000")
        self.assertEqual(aggregate[0]["final_median_r80_mean"], "0.150000")

    def test_sample_pair_bgr_uniform_radius_uses_state_priority_and_uniform_radius(self) -> None:
        class FakeBench:
            pass

        records = [
            LevelRecord(
                id="low",
                domain="test",
                task_id="inverted_pendulum",
                clean_success_hat=1.0,
                feasibility_hat=1.0,
                r_alpha_hat=0.02,
                sharpness_hat=0.1,
                uncertainty_hat=0.1,
            ),
            LevelRecord(
                id="target",
                domain="test",
                task_id="inverted_pendulum",
                clean_success_hat=1.0,
                feasibility_hat=1.0,
                r_alpha_hat=0.12,
                sharpness_hat=1.0,
                uncertainty_hat=1.0,
            ),
        ]
        args = Namespace(
            max_radius=0.30,
            priority_temperature=0.5,
            uniform_mix=0.0,
            target_radius=0.12,
            radius_bandwidth=0.05,
            radius_noise=0.01,
            radius_uniform_mix=0.0,
        )
        scorer = type(
            "Scorer",
            (),
            {"score": lambda _self, record, _step: 100.0 if record.id == "target" else 1e-6},
        )()

        replay_idx, sigma = sample_pair(
            "bgr_uniform_radius",
            FakeBench(),
            records,
            scorer,
            np.random.default_rng(7),
            args,
            0,
        )

        self.assertEqual(replay_idx, 1)
        self.assertGreaterEqual(sigma, 0.0)
        self.assertLessEqual(sigma, 0.30)


if __name__ == "__main__":
    unittest.main()
