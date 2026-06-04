import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "queue_openvla_oft_goal_adapt.sh"
PREREG_SCRIPT = ROOT / "scripts" / "queue_openvla_oft_preregistered_goal_adapt.sh"


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
                "REMOTE_RUN_ROOT": "/tmp/bgr-runs",
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

    def run_preregistered(self, *args: str, check: bool = True, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "ADAPT_METHODS": "bgr",
                "PERTURB_METHODS": "official",
                "REMOTE_PROJECT": "/tmp/bgr",
                "REMOTE_LOG_DIR": "/tmp/bgr-logs",
                "REMOTE_RUN_ROOT": "/tmp/bgr-runs",
                "REMOTE_HF_HOME": "/tmp/hf-home",
                "REMOTE_TRANSFORMERS_CACHE": "/tmp/hf-home/hub",
                "OPENVLA_OFT_ROOT": "/tmp/openvla-oft",
                "OPENVLA_OFT_TORCHRUN": "/tmp/openvla-oft/.venv/bin/torchrun",
                "OPENVLA_OFT_PY": "/tmp/openvla-oft/.venv/bin/python",
                "OPENVLA_OFT_SITE": "/tmp/openvla-oft/.venv/lib/python3.10/site-packages",
                "LIBERO_ROOT": "/tmp/LIBERO",
                "OFFICIAL_STATS": "/tmp/official/dataset_statistics.json",
                "GIT_PULL": "0",
            }
        )
        env.update(overrides)
        return subprocess.run(
            ["bash", str(PREREG_SCRIPT), *args],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

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

    def test_dry_run_uses_remote_run_root_for_eval_logs(self) -> None:
        output = self.run_dry()
        self.assertIn('mkdir -p "/tmp/bgr-runs/unit_artifact/logs/bgr"', output)
        self.assertIn('--local_log_dir "/tmp/bgr-runs/unit_artifact/logs/bgr"', output)
        self.assertIn('BGR_EVAL_ROLLOUT_DIR="/tmp/bgr-runs/unit_artifact/logs/bgr/rollouts"', output)

    def test_preregistered_p4096_adapt_dry_run_pins_recipe(self) -> None:
        result = self.run_preregistered("--adapt-only")
        output = result.stdout
        self.assertIn("cleanmix_p4096_commonavail_step50500_lr5em7_identitylora_imageaug_officialtrainstats_prereg_v1", output)
        self.assertIn("/tmp/bgr-runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p4096_commonavail_v1", output)
        self.assertIn('BGR_TRAIN_DATASET_STATISTICS_SOURCE="/tmp/official/dataset_statistics.json"', output)
        self.assertIn('--learning_rate "5e-7"', output)
        self.assertIn('--max_steps "50500"', output)
        self.assertIn('--image_aug "True"', output)
        self.assertIn('--num_tasks "10"', output)
        self.assertIn('--num_trials_per_task "10"', output)

    def test_preregistered_p4096_perturb_dry_run_pins_full_goal_eval(self) -> None:
        result = self.run_preregistered("--perturb-only", PERTURB_METHODS="bgr")
        output = result.stdout
        self.assertIn("fullgoal10x10", output)
        self.assertIn("Promotion gate: BGR must beat random and official by >=10/400", output)
        self.assertIn('--num_tasks "10"', output)
        self.assertIn('--num_trials_per_task "10"', output)
        self.assertIn("/tmp/bgr-runs/openvla_oft_goal_adapt_bgr_cleanmix_p4096_commonavail_step50500", output)

    def test_preregistered_p4096_refuses_perturb_submit_without_merge_dependencies(self) -> None:
        result = self.run_preregistered("--perturb-only", "--submit-perturb", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing --submit-perturb without BGR_DEPENDENCY and RANDOM_DEPENDENCY", result.stderr)


if __name__ == "__main__":
    unittest.main()
