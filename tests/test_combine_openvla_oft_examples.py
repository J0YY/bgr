import json
import tempfile
import unittest
from pathlib import Path

from scripts.combine_openvla_oft_examples import combine_sources


class CombineOpenVLAOFTExamplesTest(unittest.TestCase):
    def test_combines_sources_and_rewrites_asset_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            for subdir in ["images", "wrist_images", "arrays"]:
                (source / subdir).mkdir(parents=True, exist_ok=True)
            (source / "images" / "example_00000.png").write_bytes(b"image")
            (source / "wrist_images" / "example_00000.png").write_bytes(b"wrist")
            (source / "arrays" / "example_00000.npz").write_bytes(b"array")
            row = {
                "image": "images/example_00000.png",
                "wrist_image": "wrist_images/example_00000.png",
                "array": "arrays/example_00000.npz",
                "perturbation_type": "identity",
            }
            (source / "examples.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            out = root / "combined"
            rows = combine_sources([("clean", source)], out)

            self.assertEqual(rows[0]["image"], "clean/images/example_00000.png")
            self.assertEqual(rows[0]["mix_source"], "clean")
            self.assertTrue((out / "clean" / "images" / "example_00000.png").exists())


if __name__ == "__main__":
    unittest.main()
