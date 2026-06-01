import unittest

import numpy as np

from bgr.libero_probe import finite_observation


class LiberoProbeTest(unittest.TestCase):
    def test_finite_observation_rejects_nan_arrays(self):
        self.assertTrue(finite_observation({"image": np.zeros((2, 2)), "state": np.ones(3)}))
        self.assertFalse(finite_observation({"state": np.array([0.0, np.nan])}))


if __name__ == "__main__":
    unittest.main()
