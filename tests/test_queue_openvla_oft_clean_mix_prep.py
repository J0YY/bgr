import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "queue_openvla_oft_clean_mix_prep.sh"


class QueueOpenVLAOFTCleanMixPrepTest(unittest.TestCase):
    def run_dry(self, **overrides: str) -> str:
        env = os.environ.copy()
        env.update(
            {
                "TAG": "unit",
                "REMOTE_HOST": "example.invalid",
                "REMOTE_PROJECT": "/tmp/bgr",
                "REMOTE_LOG_DIR": "/tmp/bgr-logs",
                "REMOTE_RUN_ROOT": "/tmp/bgr-runs",
                "OPENVLA_OFT_ROOT": "/tmp/openvla-oft",
                "OPENVLA_OFT_PY": "/tmp/openvla-oft/.venv/bin/python",
                "OPENVLA_OFT_SITE": "/tmp/openvla-oft/.venv/lib/python3.10/site-packages",
                "LIBERO_ROOT": "/tmp/LIBERO",
                "SYNC_CODE": "0",
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

    def test_perturb_repeat_creates_distinct_mix_sources(self) -> None:
        output = self.run_dry(PERTURB_REPEAT="2", SOURCE_ARTIFACT_ROOT="/tmp/source-artifacts")

        self.assertIn('/tmp/source-artifacts/libero_openvla_observation_proposal_balanced_expfit_seed1_lp2_h160', output)
        self.assertIn('/tmp/source-artifacts/libero_openvla_observation_random_balanced_seed1b_skip_lp2_h160', output)
        self.assertIn('/tmp/bgr-runs/openvla_teacher_oft_bgr_cleanmix_unit_v1', output)
        self.assertIn('/tmp/bgr-runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_unit_v1', output)
        self.assertIn('--source "perturb_${repeat_idx}=', output)
        self.assertIn('for repeat_idx in $(seq 1 "2"); do', output)
        self.assertIn('BGR_COMBINE_ARGS+=(--source "perturb_${repeat_idx}=', output)
        self.assertIn('RANDOM_COMBINE_ARGS+=(--source "perturb_${repeat_idx}=', output)


if __name__ == "__main__":
    unittest.main()
