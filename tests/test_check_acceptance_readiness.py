import tempfile
import unittest
from pathlib import Path

from scripts.check_acceptance_readiness import OPENVLA_WEIGHTED_AVAILABLE
from scripts.check_acceptance_readiness import learned_policy_gate


def _write_weighted_summary(root: Path) -> None:
    path = root / OPENVLA_WEIGHTED_AVAILABLE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "method,perturbation,episodes,successes,success_rate",
                "bgr,identity,100,99,0.99",
                "bgr,blur,100,98,0.98",
                "bgr,brightness,100,99,0.99",
                "bgr,occlusion,100,75,0.75",
                "bgr,shift,100,95,0.95",
                "official,identity,100,99,0.99",
                "official,blur,100,97,0.97",
                "official,brightness,100,98,0.98",
                "official,occlusion,100,74,0.74",
                "official,shift,100,98,0.98",
                "random,identity,100,99,0.99",
                "random,blur,100,99,0.99",
                "random,brightness,100,99,0.99",
                "random,occlusion,100,75,0.75",
                "",
            ]
        ),
        encoding="utf-8",
    )


class CheckAcceptanceReadinessTest(unittest.TestCase):
    def test_learned_policy_gate_prefers_latest_weighted_official_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_weighted_summary(root)

            gate = learned_policy_gate(root)

        self.assertFalse(gate.passed)
        self.assertEqual(gate.name, "learned-policy OpenVLA/LIBERO")
        self.assertIn("latest weighted audit", gate.detail)
        self.assertIn("BGR 367/400", gate.detail)
        self.assertIn("official 367/400", gate.detail)
        self.assertIn("random 273/300 available rows", gate.detail)
        self.assertIn("official_margin=0", gate.detail)


if __name__ == "__main__":
    unittest.main()
