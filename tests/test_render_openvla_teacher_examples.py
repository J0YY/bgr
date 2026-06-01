import unittest

import numpy as np

from scripts.render_openvla_teacher_examples import _apply_perturbation, _keep_row, _libero_oft_state, _suite_name


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
