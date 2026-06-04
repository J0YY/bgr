import os
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_openvla_oft_eval import _summarize_method


class OpenVLAOFTEvalSummaryTest(unittest.TestCase):
    def test_summarize_latest_eval_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "old.txt").write_text(
                "\n".join(
                    [
                        "Current task success rate: 0.0",
                        "Final results:",
                        "Total episodes: 1",
                        "Total successes: 0",
                        "Overall success rate: 0.0000 (0.0%)",
                    ]
                ),
                encoding="utf-8",
            )
            latest = log_dir / "latest.txt"
            latest.write_text(
                "\n".join(
                    [
                        "Current task success rate: 0.3333333333333333",
                        "Current task success rate: 0.6666666666666666",
                        "Final results:",
                        "Total episodes: 6",
                        "Total successes: 3",
                        "Overall success rate: 0.5000 (50.0%)",
                    ]
                ),
                encoding="utf-8",
            )

            row = _summarize_method("bgr", log_dir)

        self.assertEqual(row["method"], "bgr")
        self.assertEqual(row["episodes"], 6)
        self.assertEqual(row["successes"], 3)
        self.assertEqual(row["success_rate"], 0.5)
        self.assertEqual(row["num_tasks"], 2)
        self.assertIn("0.3333333333333333", row["task_success_rates"])

    def test_uses_newest_complete_log_when_latest_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            complete = log_dir / "complete.txt"
            complete.write_text(
                "\n".join(
                    [
                        "Current task success rate: 0.3333333333333333",
                        "Final results:",
                        "Total episodes: 15",
                        "Total successes: 14",
                        "Overall success rate: 0.9333 (93.3%)",
                    ]
                ),
                encoding="utf-8",
            )
            partial = log_dir / "partial.txt"
            partial.write_text("Current task success rate: 0.3333333333333333\n", encoding="utf-8")
            os.utime(complete, (1, 1))
            os.utime(partial, (2, 2))

            row = _summarize_method("bgr", log_dir)

        self.assertEqual(row["log"], str(complete))
        self.assertEqual(row["episodes"], 15)
        self.assertEqual(row["successes"], 14)


if __name__ == "__main__":
    unittest.main()
