import unittest

import numpy as np

from tools.fetch_object_goal_recovery_calibration import controller_action


class FetchObjectGoalRecoveryCalibrationTest(unittest.TestCase):
    def test_scripted_push_far_approaches_from_farther_behind_object(self) -> None:
        obs = {
            "observation": np.array([1.10, 1.00, 0.42, 1.20, 1.00, 0.42], dtype=float),
            "achieved_goal": np.array([1.20, 1.00, 0.42], dtype=float),
            "desired_goal": np.array([1.40, 1.00, 0.42], dtype=float),
        }

        near_action = controller_action("scripted_push", obs, threshold=0.05, gain=10.0)
        far_action = controller_action("scripted_push_far", obs, threshold=0.05, gain=10.0)

        self.assertLess(far_action[0], near_action[0])
        self.assertGreater(far_action[2], near_action[2])
        self.assertEqual(far_action[1], near_action[1])

    def test_controller_returns_zero_when_goal_is_already_reached(self) -> None:
        obs = {
            "observation": np.array([1.00, 1.00, 0.42, 1.20, 1.00, 0.42], dtype=float),
            "achieved_goal": np.array([1.20, 1.00, 0.42], dtype=float),
            "desired_goal": np.array([1.21, 1.00, 0.42], dtype=float),
        }

        action = controller_action("scripted_push_far", obs, threshold=0.05, gain=10.0)

        np.testing.assert_allclose(action, np.zeros(4, dtype=float))


if __name__ == "__main__":
    unittest.main()
