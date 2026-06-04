import unittest
from pathlib import Path


class SubmissionFramingTest(unittest.TestCase):
    def test_submission_facing_files_use_boundary_guided_replay(self):
        root = Path(__file__).resolve().parents[1]
        checked_files = [
            root / "README.md",
            root / "pyproject.toml",
            root / "spec.md",
            root / "src" / "bgr" / "__init__.py",
            root / "paper" / "README.md",
            root / "paper" / "main.tex",
            root / "paper" / "ReproducibilityChecklist.tex",
        ]

        for path in checked_files:
            with self.subTest(path=path.relative_to(root)):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("Bifurcation", text)
                self.assertNotIn("bifurcation", text)


if __name__ == "__main__":
    unittest.main()
