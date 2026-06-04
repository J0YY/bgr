import tempfile
import unittest
from pathlib import Path

from scripts.summarize_openvla_oft_perturb_eval import aggregate_by_method, summarize_perturbation_logs


def _write_log(path: Path, episodes: int, successes: int, rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "Current task success rate: 0.3333333333333333",
                "Final results:",
                f"Total episodes: {episodes}",
                f"Total successes: {successes}",
                f"Overall success rate: {rate:.4f} ({rate * 100:.1f}%)",
            ]
        ),
        encoding="utf-8",
    )


def _write_partial_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Current task success rate: 0.3333333333333333\n", encoding="utf-8")


class OpenVLAOFTPerturbEvalSummaryTest(unittest.TestCase):
    def test_summarize_nested_perturbation_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_log(root / "bgr" / "identity" / "eval.txt", 15, 14, 14 / 15)
            _write_log(root / "bgr" / "blur" / "eval.txt", 15, 9, 9 / 15)
            _write_log(root / "random" / "identity" / "eval.txt", 15, 13, 13 / 15)

            rows = summarize_perturbation_logs(root)

        self.assertEqual([(row["method"], row["perturbation"]) for row in rows], [("bgr", "identity"), ("bgr", "blur"), ("random", "identity")])
        self.assertEqual(rows[1]["episodes"], 15)
        self.assertEqual(rows[1]["successes"], 9)
        self.assertAlmostEqual(rows[1]["success_rate"], 0.6)

    def test_skips_incomplete_perturbation_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_log(root / "bgr" / "identity" / "eval.txt", 15, 14, 14 / 15)
            _write_partial_log(root / "random" / "identity" / "eval.txt")
            (root / "random" / "blur").mkdir(parents=True)

            rows = summarize_perturbation_logs(root)

        self.assertEqual([(row["method"], row["perturbation"]) for row in rows], [("bgr", "identity")])

    def test_aggregate_excludes_identity_from_perturbed_mean(self):
        rows = [
            {"method": "bgr", "perturbation": "identity", "episodes": 15, "successes": 14, "success_rate": 0.9333},
            {"method": "bgr", "perturbation": "blur", "episodes": 15, "successes": 9, "success_rate": 0.6000},
            {"method": "bgr", "perturbation": "shift", "episodes": 15, "successes": 6, "success_rate": 0.4000},
        ]

        aggregate = aggregate_by_method(rows)

        self.assertEqual(aggregate[0]["method"], "bgr")
        self.assertAlmostEqual(aggregate[0]["identity_success_rate"], 14 / 15)
        self.assertAlmostEqual(aggregate[0]["mean_perturbed_success_rate"], 0.5)
        self.assertEqual(aggregate[0]["num_perturbed"], 2)


if __name__ == "__main__":
    unittest.main()
