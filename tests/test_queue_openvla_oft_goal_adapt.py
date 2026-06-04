import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "queue_openvla_oft_goal_adapt.sh"


class QueueOpenVlaOftGoalAdaptTest(unittest.TestCase):
    def run_dry(self, **overrides: str) -> str:
        env = os.environ.copy()
        env.update(
            {
                "METHODS": "bgr",
                "TAG": "unit",
                "EVAL_ARTIFACT": "unit_artifact",
                "REMOTE_PROJECT": "/tmp/bgr",
                "REMOTE_LOG_DIR": "/tmp/bgr-logs",
                "REMOTE_HF_HOME": "/tmp/hf-home",
                "REMOTE_TRANSFORMERS_CACHE": "/tmp/hf-home/hub",
                "OPENVLA_OFT_ROOT": "/tmp/openvla-oft",
                "OPENVLA_OFT_TORCHRUN": "/tmp/openvla-oft/.venv/bin/torchrun",
                "OPENVLA_OFT_PY": "/tmp/openvla-oft/.venv/bin/python",
                "OPENVLA_OFT_SITE": "/tmp/openvla-oft/.venv/lib/python3.10/site-packages",
                "LIBERO_ROOT": "/tmp/LIBERO",
                "BGR_DATA_ROOT": "/tmp/bgr-data",
                "BGR_RUN_ROOT": "/tmp/bgr-run",
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

    def test_dry_run_defaults_image_augmentation_off(self) -> None:
        output = self.run_dry()
        self.assertIn('--image_aug "False"', output)

    def test_dry_run_can_enable_image_augmentation(self) -> None:
        output = self.run_dry(IMAGE_AUG="True")
        self.assertIn('--image_aug "True"', output)

    def test_dry_run_uses_remote_cache_overrides(self) -> None:
        output = self.run_dry()
        self.assertIn('HF_HOME="/tmp/hf-home"', output)
        self.assertIn('TRANSFORMERS_CACHE="/tmp/hf-home/hub"', output)


if __name__ == "__main__":
    unittest.main()
