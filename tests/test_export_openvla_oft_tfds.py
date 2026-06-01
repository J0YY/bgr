import unittest

import numpy as np

from scripts.export_openvla_oft_tfds import _episode_records, _summary


def _record(candidate_name: str, perturbation_type: str) -> dict:
    return {
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.arange(8, dtype=np.float32),
        "action": np.arange(7, dtype=np.float32),
        "language": "open the drawer",
        "metadata": {
            "array": f"arrays/{candidate_name}.npz",
            "suite": "goal",
            "task_idx": 0,
            "task_name": "open_drawer",
            "candidate_name": candidate_name,
            "perturbation_type": perturbation_type,
            "instruction": "open the drawer",
        },
    }


class ExportOpenVLAOFTTFDSTest(unittest.TestCase):
    def test_episode_records_keep_rlds_step_fields(self):
        episodes = _episode_records([_record("blur", "blur"), _record("shift", "shift")])

        self.assertEqual([key for key, _example in episodes], ["episode_00000", "episode_00001"])
        step = episodes[0][1]["steps"][0]
        self.assertEqual(step["observation"]["image"].shape, (224, 224, 3))
        self.assertEqual(step["observation"]["state"].shape, (8,))
        self.assertEqual(step["action"].shape, (7,))
        self.assertTrue(step["is_first"])
        self.assertTrue(step["is_last"])
        self.assertTrue(step["is_terminal"])
        self.assertEqual(episodes[0][1]["episode_metadata"]["task_name"], "open_drawer")

    def test_summary_counts_steps_and_families(self):
        episodes = _episode_records([_record("blur", "blur"), _record("shift", "shift")])
        summary = _summary("bgr_test", "1.0.0", "tmp/bgr_test/1.0.0", episodes)

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["perturbation_types"], ["blur", "shift"])
        self.assertEqual(summary["dataset_name"], "bgr_test")


if __name__ == "__main__":
    unittest.main()
