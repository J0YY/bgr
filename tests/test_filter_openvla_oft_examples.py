from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.filter_openvla_oft_examples import filter_examples


class FilterOpenVlaOftExamplesTest(unittest.TestCase):
    def test_filters_by_family_cap_and_copies_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            out = root / "out"
            rows = [
                self.write_example(source, 0, "blur"),
                self.write_example(source, 1, "occlusion"),
                self.write_example(source, 2, "occlusion"),
                self.write_example(source, 3, "shift"),
            ]
            (source / "examples.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            kept = filter_examples(source, out, {"occlusion": 1})

            self.assertEqual([row["perturbation_type"] for row in kept], ["blur", "occlusion", "shift"])
            self.assertTrue((out / "images" / "example_00000.png").exists())
            self.assertTrue((out / "images" / "example_00001.png").exists())
            self.assertFalse((out / "images" / "example_00002.png").exists())
            self.assertTrue((out / "arrays" / "example_00003.npz").exists())

    def write_example(self, source: Path, index: int, perturbation_type: str) -> dict[str, str]:
        image = Path("images") / f"example_{index:05d}.png"
        wrist = Path("wrist_images") / f"example_{index:05d}.png"
        array = Path("arrays") / f"example_{index:05d}.npz"
        for relative in [image, wrist, array]:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"asset {index}".encode("utf-8"))
        return {
            "image": image.as_posix(),
            "wrist_image": wrist.as_posix(),
            "array": array.as_posix(),
            "perturbation_type": perturbation_type,
        }


if __name__ == "__main__":
    unittest.main()
