import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.render_openvla_teacher_examples import (
    _apply_perturbation,
    _keep_row,
    _load_rows,
    _libero_oft_state,
    _select_balanced_episode_rows,
    _suite_name,
)


class RenderOpenVLATeacherExamplesTest(unittest.TestCase):
    def test_suite_name_canonicalizes_short_names(self):
        self.assertEqual(_suite_name("object"), "libero_object")
        self.assertEqual(_suite_name("libero_goal"), "libero_goal")

    def test_occlusion_changes_center_pixels(self):
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        out = _apply_perturbation(image, "occlusion", {"fraction": 0.4})
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(int(out[5, 5, 0]), 0)
        self.assertEqual(int(out[0, 0, 0]), 255)

    def test_first_per_family_selection(self):
        seen = set()
        self.assertTrue(_keep_row({"perturbation_type": "blur"}, "first_per_family", seen))
        self.assertFalse(_keep_row({"perturbation_type": "blur"}, "first_per_family", seen))
        self.assertTrue(_keep_row({"perturbation_type": "shift"}, "first_per_family", seen))

    def test_balanced_episode_selection_keeps_contiguous_steps(self):
        rows = []
        for episode_idx, family in enumerate(["blur", "shift", "blur"]):
            candidate = f"{family}_{episode_idx}"
            for step_idx in range(3):
                rows.append(
                    {
                        "suite": "goal",
                        "task_idx": 0,
                        "task_name": "open_drawer",
                        "episode_idx": episode_idx,
                        "init_state_idx": 0,
                        "candidate_name": candidate,
                        "perturbation_type": family,
                        "step_idx": step_idx,
                    }
                )

        selected = _select_balanced_episode_rows(
            rows,
            max_examples=4,
            episodes_per_family=1,
            max_steps_per_episode=2,
        )

        self.assertEqual([row["perturbation_type"] for row in selected], ["blur", "blur", "shift", "shift"])
        self.assertEqual([row["step_idx"] for row in selected], [0, 1, 0, 1])

    def test_load_rows_filters_by_method_before_balancing(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            rows = []
            for method in ["bgr_boundary", "random_balanced"]:
                for family in ["blur", "shift"]:
                    for step_idx in range(2):
                        rows.append(
                            {
                                "method": method,
                                "suite": "goal",
                                "task_idx": 0,
                                "task_name": "open_drawer",
                                "episode_idx": 0,
                                "init_state_idx": 0,
                                "candidate_name": f"{method}_{family}",
                                "perturbation_type": family,
                                "step_idx": step_idx,
                            }
                        )
            path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n", encoding="utf-8")

            selected = _load_rows(
                path,
                4,
                "balanced_episodes",
                methods=("random_balanced",),
                episodes_per_family=1,
                max_steps_per_episode=2,
            )

            self.assertEqual({row["method"] for row in selected}, {"random_balanced"})
            self.assertEqual([row["perturbation_type"] for row in selected], ["blur", "blur", "shift", "shift"])

    def test_libero_oft_state_shape(self):
        obs = {
            "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.1, 0.2]),
        }
        state = _libero_oft_state(obs)
        self.assertEqual(state.shape, (8,))
        np.testing.assert_allclose(state[:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(state[3:6], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(state[-2:], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
