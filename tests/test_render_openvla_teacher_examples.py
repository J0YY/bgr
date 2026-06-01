import unittest

import numpy as np

from scripts.render_openvla_teacher_examples import _apply_perturbation, _suite_name


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


if __name__ == "__main__":
    unittest.main()
