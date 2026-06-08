import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bgr.records import LevelRecord  # noqa: E402
from tools.bsuite_cartpole_recovery_probe import (  # noqa: E402
    CartpoleRecoveryProbe,
    CartpoleReplayState,
    build_parser,
    centered_angle,
    main,
    sample_training_pair,
)


def bsuite_available() -> bool:
    return importlib.util.find_spec("bsuite") is not None


def tiny_args(**overrides):
    args = build_parser().parse_args([])
    for key, value in {
        "out": "runs/bsuite_cartpole_test",
        "seeds": "0",
        "methods": "uniform,bgr",
        "iterations": 1,
        "eval_every": 1,
        "train_batch_size": 1,
        "replay_states": 4,
        "max_steps": 3,
        "eval_grid_size": 3,
        "policy_init_steps": 4,
        "refresh_per_eval": 2,
        "baseline_candidates": 2,
        **overrides,
    }.items():
        setattr(args, key, value)
    return args


class BsuiteCartpoleRecoveryProbeTest(unittest.TestCase):
    def test_centered_angle_wraps_to_signed_pi_interval(self) -> None:
        self.assertAlmostEqual(centered_angle(0.0), 0.0)
        self.assertAlmostEqual(centered_angle(2.0 * np.pi), 0.0)
        self.assertGreaterEqual(centered_angle(3.0 * np.pi), -np.pi)
        self.assertLessEqual(centered_angle(3.0 * np.pi), np.pi)

    def test_sample_pair_bgr_uniform_radius_uses_state_priority_and_uniform_radius(self) -> None:
        class FakeBench:
            pass

        records = [
            LevelRecord(
                id="low",
                domain="test",
                task_id="bsuite_cartpole",
                clean_success_hat=1.0,
                feasibility_hat=1.0,
                r_alpha_hat=0.05,
                sharpness_hat=0.1,
                uncertainty_hat=0.1,
            ),
            LevelRecord(
                id="target",
                domain="test",
                task_id="bsuite_cartpole",
                clean_success_hat=1.0,
                feasibility_hat=1.0,
                r_alpha_hat=0.45,
                sharpness_hat=1.0,
                uncertainty_hat=1.0,
            ),
        ]
        args = Namespace(
            priority_temperature=0.5,
            uniform_mix=0.0,
            radius_noise=0.08,
            radius_uniform_mix=0.0,
        )
        scorer = type(
            "Scorer",
            (),
            {"score": lambda _self, record, _step: 100.0 if record.id == "target" else 1e-6},
        )()

        replay_idx, sigma = sample_training_pair(
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
        self.assertLessEqual(sigma, 1.0)

    @unittest.skipUnless(bsuite_available(), "bsuite optional dependency is not installed")
    def test_package_backed_rollout_and_perturbations_are_feasible(self) -> None:
        bench = CartpoleRecoveryProbe(seed=0, args=tiny_args())
        try:
            self.assertEqual(len(bench.states), 4)
            for state in bench.evaluation_starts(0, 1.0):
                self.assertGreater(np.cos(state.theta), bench.height_threshold)
                self.assertLess(abs(state.x), bench.x_threshold)

            near_upright = CartpoleReplayState(x=0.0, x_dot=0.0, theta=0.01, theta_dot=0.0)
            self.assertIsInstance(bench.rollout_state(near_upright, train=False), bool)
        finally:
            bench.close()

    @unittest.skipUnless(bsuite_available(), "bsuite optional dependency is not installed")
    def test_tiny_cli_writes_compact_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_argv = sys.argv
            sys.argv = [
                "bsuite_cartpole_recovery_probe.py",
                "--out",
                tmp,
                "--seeds",
                "0",
                "--methods",
                "uniform,bgr",
                "--iterations",
                "1",
                "--eval-every",
                "1",
                "--train-batch-size",
                "1",
                "--replay-states",
                "4",
                "--max-steps",
                "3",
                "--eval-grid-size",
                "3",
                "--policy-init-steps",
                "4",
                "--refresh-per-eval",
                "2",
            ]
            try:
                main()
            finally:
                sys.argv = old_argv

            out = Path(tmp)
            self.assertTrue((out / "summary.csv").exists())
            self.assertTrue((out / "package_versions.json").exists())
            self.assertIn("method,seed,final_clean", (out / "summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
