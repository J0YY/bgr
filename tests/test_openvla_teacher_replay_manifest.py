import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_openvla_teacher_replay_manifest import _export_dir


class OpenVLATeacherReplayManifestTest(unittest.TestCase):
    def test_exports_successful_native_actions_for_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validate = root / "validate"
            validate.mkdir()
            (root / "top_k_candidates.json").write_text(
                json.dumps(
                    [
                        {
                            "name": "cand_blur",
                            "spec": "cand_blur:blur:1.0",
                            "perturbation_type": "blur",
                            "perturbation_params": {"radius": 1.0},
                            "observed_cf_rate": 0.5,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "suite": "object",
                    "task_idx": 0,
                    "task_name": "task",
                    "episode_idx": 0,
                    "init_state_idx": 0,
                    "perturbation_type": "identity",
                    "perturbation_name": "native",
                    "step_idx": 0,
                    "instruction": "do task",
                    "executed_action": [0.1, 0.2],
                    "token_ids": [1, 2],
                    "success_after_step": False,
                },
                {
                    "suite": "object",
                    "task_idx": 0,
                    "task_name": "task",
                    "episode_idx": 0,
                    "init_state_idx": 0,
                    "perturbation_type": "identity",
                    "perturbation_name": "native",
                    "step_idx": 1,
                    "instruction": "do task",
                    "executed_action": [0.3, 0.4],
                    "token_ids": [3, 4],
                    "success_after_step": True,
                },
                {
                    "suite": "object",
                    "task_idx": 0,
                    "task_name": "task",
                    "episode_idx": 0,
                    "init_state_idx": 0,
                    "perturbation_type": "blur",
                    "perturbation_name": "cand_blur",
                    "step_idx": 0,
                    "executed_action": [9.0, 9.0],
                    "success_after_step": False,
                },
            ]
            (validate / "observation_steps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = _export_dir(root, "bgr_boundary", 0.25, 0.75, max_steps_per_episode=8, boundary_only=True)
            self.assertEqual(len(output), 2)
            self.assertEqual(output[0]["target_action"], "[0.1, 0.2]")
            self.assertTrue(output[0]["in_boundary_band"])

    def test_success_can_occur_after_export_step_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validate = root / "validate"
            validate.mkdir()
            (root / "top_k_candidates.json").write_text(
                json.dumps([{"name": "cand_blur", "spec": "cand_blur:blur:1.0", "observed_cf_rate": 0.5}]),
                encoding="utf-8",
            )
            rows = [
                {
                    "suite": "object",
                    "task_idx": 0,
                    "task_name": "task",
                    "episode_idx": 0,
                    "init_state_idx": 0,
                    "perturbation_type": "identity",
                    "perturbation_name": "native",
                    "step_idx": 0,
                    "instruction": "do task",
                    "executed_action": [0.1],
                    "success_after_step": False,
                },
                {
                    "suite": "object",
                    "task_idx": 0,
                    "task_name": "task",
                    "episode_idx": 0,
                    "init_state_idx": 0,
                    "perturbation_type": "identity",
                    "perturbation_name": "native",
                    "step_idx": 1,
                    "instruction": "do task",
                    "executed_action": [0.2],
                    "success_after_step": True,
                },
                {
                    "suite": "object",
                    "task_idx": 0,
                    "task_name": "task",
                    "episode_idx": 0,
                    "init_state_idx": 0,
                    "perturbation_type": "blur",
                    "perturbation_name": "cand_blur",
                    "step_idx": 0,
                    "executed_action": [9.0],
                    "success_after_step": False,
                },
            ]
            (validate / "observation_steps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = _export_dir(root, "bgr_boundary", 0.25, 0.75, max_steps_per_episode=1, boundary_only=True)
            self.assertEqual(len(output), 1)
            self.assertEqual(output[0]["step_idx"], 0)


if __name__ == "__main__":
    unittest.main()
