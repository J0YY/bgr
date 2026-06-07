import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "queue_openvla_oft_goal_adapt.sh"
PREREG_SCRIPT = ROOT / "scripts" / "queue_openvla_oft_preregistered_goal_adapt.sh"
WEIGHTED_PREREG_SCRIPT = ROOT / "scripts" / "queue_openvla_oft_preregistered_weighted_perturb.sh"
PROXIMAL_PREREG_SCRIPT = ROOT / "scripts" / "queue_openvla_oft_preregistered_proximal_anchor.sh"
PROXIMAL_SYNC_SCRIPT = ROOT / "scripts" / "sync_openvla_oft_proximal_anchor_results.sh"
WEIGHTED_SYNC_SCRIPT = ROOT / "scripts" / "sync_openvla_oft_weighted_perturb_results.sh"


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

    def run_weighted_preregistered(
        self, *args: str, check: bool = True, **overrides: str
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
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
                "SYNC_CODE": "0",
            }
        )
        env.update(overrides)
        return subprocess.run(
            ["bash", str(WEIGHTED_PREREG_SCRIPT), *args],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def run_proximal_preregistered(
        self, *args: str, check: bool = True, **overrides: str
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
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
                "SYNC_CODE": "0",
            }
        )
        env.update(overrides)
        return subprocess.run(
            ["bash", str(PROXIMAL_PREREG_SCRIPT), *args],
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

    def test_goal_adapt_dry_run_can_enable_proximal_anchor(self) -> None:
        output = self.run_dry(
            PROXIMAL_ANCHOR_L2="1.0",
            TRAIN_DATASET_STATISTICS_SOURCE="/tmp/official/dataset_statistics.json",
        )
        self.assertIn('export BGR_PROXIMAL_ANCHOR_L2="1.0"', output)
        self.assertIn("Enabling official-checkpoint proximal anchor", output)
        self.assertIn("proximal_anchor_l2", output)
        self.assertIn("bgr_anchor_params", output)
        self.assertIn("with torch.no_grad():", output)
        self.assertIn("normalized_loss.backward()", output)
        self.assertIn("param.grad.add_", output)
        self.assertNotIn("loss = loss + bgr_proximal_anchor_l2 * proximal_anchor_loss", output)

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

    def test_weighted_preregistered_prep_dry_run_pins_perturb_repeat(self) -> None:
        result = self.run_weighted_preregistered("--prep-only")
        output = result.stdout
        self.assertIn("weighted perturbation curriculum", output)
        self.assertIn("PREP_TAG=p2048unique_perturbrepeat3_prereg", output)
        self.assertIn("PERTURB_REPEAT=3", output)
        self.assertIn('Rendering BGR perturbation subset', output)
        self.assertIn('--max-examples "2048"', output)
        self.assertIn('--episodes-per-family "8"', output)
        self.assertIn('for repeat_idx in $(seq 1 "3"); do', output)

    def test_weighted_preregistered_adapt_dry_run_uses_weighted_roots(self) -> None:
        result = self.run_weighted_preregistered("--adapt-only")
        output = result.stdout
        self.assertIn("cleanmix_p2048unique_perturbrepeat3_prereg_step50500", output)
        self.assertIn("/tmp/bgr-runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p2048unique_perturbrepeat3_prereg_v1", output)
        self.assertIn("/tmp/bgr-runs/openvla_oft_tfds_libero_goal_random_cleanmix_p2048unique_perturbrepeat3_prereg_v1", output)
        self.assertIn('BGR_TRAIN_DATASET_STATISTICS_SOURCE="/tmp/official/dataset_statistics.json"', output)
        self.assertIn('--learning_rate "5e-7"', output)
        self.assertIn('--image_aug "True"', output)

    def test_weighted_preregistered_refuses_perturb_submit_without_merge_dependencies(self) -> None:
        result = self.run_weighted_preregistered("--perturb-only", "--submit-perturb", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing --submit-perturb without BGR_DEPENDENCY and RANDOM_DEPENDENCY", result.stderr)

    def test_proximal_preregistered_adapt_dry_run_pins_objective(self) -> None:
        result = self.run_proximal_preregistered("--adapt-only")
        output = result.stdout
        self.assertIn("proximal-anchor adaptation", output)
        self.assertIn("PROXIMAL_ANCHOR_L2=1.0", output)
        self.assertIn("proxanchor_l2_1em0", output)
        self.assertIn("/tmp/bgr-runs/openvla_oft_tfds_libero_goal_bgr_cleanmix_p2048unique_perturbrepeat3_prereg_v1", output)
        self.assertIn("/tmp/bgr-runs/openvla_oft_tfds_libero_goal_random_cleanmix_p2048unique_perturbrepeat3_prereg_v1", output)
        self.assertIn('export BGR_PROXIMAL_ANCHOR_L2="1.0"', output)
        self.assertIn('--learning_rate "5e-7"', output)
        self.assertIn('--image_aug "True"', output)

    def test_proximal_preregistered_refuses_perturb_submit_without_merge_dependencies(self) -> None:
        result = self.run_proximal_preregistered("--perturb-only", "--submit-perturb", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing --submit-perturb without BGR_DEPENDENCY and RANDOM_DEPENDENCY", result.stderr)

    def test_proximal_sync_dry_run_prints_fixed_paths(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "REMOTE_HOST": "athena-unit",
                "REMOTE_RUN_ROOT": "/tmp/bgr-runs",
                "LOCAL_RESULTS_ROOT": "/tmp/local-results",
            }
        )
        result = subprocess.run(
            ["bash", str(PROXIMAL_SYNC_SCRIPT), "--no-check"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        output = result.stdout
        self.assertIn("Proximal-anchor OpenVLA-OFT result sync", output)
        self.assertIn("REMOTE_HOST=athena-unit", output)
        self.assertIn("767128,767129", output)
        self.assertIn("DETAIL_JOB_IDS=767128,767134", output)
        self.assertIn("openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg_proxanchor_l2_1em0", output)
        self.assertIn("[dry-run] pass --poll", output)
        self.assertIn("[dry-run] pass --sync", output)
        self.assertIn("[skip] local gates disabled by --no-check", output)

    def test_weighted_sync_dry_run_prints_fixed_paths(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "REMOTE_HOST": "athena-unit",
                "REMOTE_RUN_ROOT": "/tmp/bgr-runs",
                "LOCAL_RESULTS_ROOT": "/tmp/local-results",
            }
        )
        result = subprocess.run(
            ["bash", str(WEIGHTED_SYNC_SCRIPT), "--no-check"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        output = result.stdout
        self.assertIn("Weighted OpenVLA-OFT perturbation result sync", output)
        self.assertIn("REMOTE_HOST=athena-unit", output)
        self.assertIn("JOB_IDS=766822,766823", output)
        self.assertIn("DETAIL_JOB_IDS=766831", output)
        self.assertIn("openvla_oft_perturb_eval_cleanmix_p2048unique_perturbrepeat3_prereg", output)
        self.assertIn("summary_available.csv", output)
        self.assertIn("[dry-run] pass --poll", output)
        self.assertIn("[dry-run] pass --sync", output)
        self.assertIn("[skip] local gates disabled by --no-check", output)


if __name__ == "__main__":
    unittest.main()
