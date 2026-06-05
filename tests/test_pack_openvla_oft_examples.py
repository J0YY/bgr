import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.pack_openvla_oft_examples import _load_record, _summary


class PackOpenVLAOFTExamplesTest(unittest.TestCase):
    def test_load_record_validates_expected_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "arrays").mkdir()
            np.savez_compressed(
                root / "arrays" / "example.npz",
                image=np.zeros((224, 224, 3), dtype=np.uint8),
                wrist_image=np.zeros((224, 224, 3), dtype=np.uint8),
                state=np.zeros((8,), dtype=np.float32),
                action=np.zeros((7,), dtype=np.float32),
                language_instruction=np.asarray("open drawer"),
            )
            row = {
                "array": "arrays/example.npz",
                "suite": "goal",
                "task_idx": 0,
                "task_name": "open_drawer",
                "candidate_name": "blur",
                "perturbation_type": "blur",
                "instruction": "open drawer",
                "mix_source": "perturb_2",
            }
            record = _load_record(root, row)
            self.assertEqual(record["image"].shape, (224, 224, 3))
            self.assertEqual(record["action"].shape, (7,))
            self.assertEqual(record["metadata"]["mix_source"], "perturb_2")
            summary = _summary([record])
            self.assertEqual(summary["families"], ["blur"])
            self.assertEqual(summary["examples"], 1)


if __name__ == "__main__":
    unittest.main()
