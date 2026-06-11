import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "queue_openvla_oft_perturb_eval.sh"


class QueueOpenVlaOftPerturbEvalTest(unittest.TestCase):
    def run_dry(self, **overrides: str) -> str:
        env = os.environ.copy()
        env.update(
            {
                "METHODS": "bgr",
                "PERTURBATIONS": 'identity={};blur={"radius":2.5}',
                "TAG": "unit",
                "EVAL_ARTIFACT": "unit_artifact",
                "REMOTE_RUN_ROOT": "/tmp/bgr-runs",
                "REMOTE_LOG_DIR": "/tmp/bgr-logs",
                "OPENVLA_OFT_ROOT": "/tmp/openvla-oft",
                "OPENVLA_OFT_PY": "/tmp/openvla-oft/.venv/bin/python",
                "OPENVLA_OFT_SITE": "/tmp/openvla-oft/.venv/lib/python3.10/site-packages",
                "BGR_CKPT": "/tmp/bgr-checkpoint",
            }
        )
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return result.stdout

    def test_dry_run_serializes_perturbations_per_method_by_default(self) -> None:
        output = self.run_dry()
        self.assertIn("bgr/identity: dry-run", output)
        self.assertIn("bgr/blur: dry-run", output)
        self.assertIn(
            'BGR_EVAL_ROLLOUT_DIR="/tmp/bgr-runs/unit_artifact/logs/bgr/identity/rollouts"',
            output,
        )
        self.assertIn('BGR_EVAL_SAVE_ROLLOUTS="0"', output)
        self.assertIn(
            "### perturb-eval-bgr-blur-unit dependency=afterok:<bgr_identity_job>",
            output,
        )

    def test_dry_run_can_disable_serial_perturbation_chain(self) -> None:
        output = self.run_dry(SERIAL_PERTURB_PER_METHOD="0")
        self.assertIn("bgr/identity: dry-run", output)
        self.assertIn("bgr/blur: dry-run", output)
        self.assertNotIn("dependency=afterok:<bgr_identity_job>", output)


if __name__ == "__main__":
    unittest.main()
