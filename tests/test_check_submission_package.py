import json
import io
import subprocess
import unittest
import tempfile
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts.check_submission_package import (
    AGGREGATE_SOURCE_ARTIFACTS,
    AUTHOR_KIT_ARTIFACTS,
    CORE_STUDY_ARTIFACTS,
    DOUBLE_BLIND_FORBIDDEN_PATTERNS,
    PdfMetadata,
    bad_log_patterns,
    check_abstract_compliance,
    check_artifact_doc_framing,
    check_aggregate_outputs_synced,
    check_aggregate_text_artifacts_synced,
    check_author_kit_artifacts,
    check_author_kit_source,
    check_checked_claim_artifact_evidence_maps,
    check_checklist_pdf,
    check_checklist_source_framing,
    check_core_study_artifacts,
    check_data_artifact_double_blind,
    check_declared_test_artifacts,
    check_double_blind_source,
    check_embedded_checklist_text,
    check_environment_artifacts,
    check_estimator_30_status,
    check_grid_margin_ablation_30_status,
    check_grid_margin_ablation_replication_status,
    check_generated_result_tables,
    check_grid_margin_full_30_status,
    check_grid_margin_learning_rate_30_status,
    check_grid_margin_regime_30_status,
    check_grid_margin_replication_status,
    check_grid_margin_stress_30_status,
    check_grid_margin_target_30_status,
    check_license_file,
    check_log,
    check_main_pdf,
    check_markdown_code_fences,
    check_manuscript_framing,
    check_openvla_bridge_scripts,
    check_p4096_status,
    check_paper_facing_openvla_staleness,
    check_paper_readme_openvla_status,
    check_paper_readme_artifact_references,
    check_paper_readme_build_provenance,
    check_paper_readme_submission_framing,
    check_paper_readme_verification_commands,
    check_pdf_font_hygiene,
    check_pdf_metadata_hygiene,
    check_pyproject_runtime_dependencies,
    check_required_artifact_scope,
    check_required_text_artifacts_double_blind,
    check_required_git_tracked,
    check_results_evidence_index,
    check_rendered_source_sync,
    check_rendered_abstract_sync,
    check_rendered_title_block_sync,
    check_results_ledger,
    check_root_readme_artifact_references,
    check_root_readme_claim_map_artifacts,
    check_root_readme_openvla_status,
    check_root_readme_command_artifacts,
    check_root_readme_submission_framing,
    check_root_readme_verification_commands,
    check_significance_text_artifacts_synced,
    check_submission_manifest,
    check_submission_manifest_docs,
    check_submission_scope_docs,
    check_suffix_coverage_full_status,
    check_suffix_coverage_full_replication_status,
    check_suffix_strategy_ablation_status,
    check_suffix_strategy_replication_status,
    check_technical_page_limit,
    check_tex_dependencies_required,
    check_title_block_compliance,
    check_toy_30_status,
    check_toy_smoke_experiment,
    container_workflow_artifact_paths,
    configured_methods,
    data_artifact_text_files,
    double_blind_leaks,
    forbidden_terms,
    main,
    missing_required_files,
    AGGREGATE_TEXT_ARTIFACTS,
    ALLOWED_TEX_LOG_ARTIFACTS,
    OPENVLA_BRIDGE_ARTIFACTS,
    OPENVLA_BRIDGE_TEST_ARTIFACTS,
    PAPER_GENERATED_VISUAL_ARTIFACTS,
    pdf_metadata,
    pdf_font_rows,
    required_submission_files,
    required_text_artifact_files,
    SIGNIFICANCE_SOURCE_ARTIFACTS,
    SIGNIFICANCE_TEXT_ARTIFACTS,
    SOURCE_ARTIFACTS,
    SUBMISSION_MANIFEST,
    TEST_ARTIFACTS,
    canonical_json,
    CHECKED_CLAIM_ARTIFACTS,
    submission_manifest_payload,
    tex_source_dependency_files,
    untracked_required_files,
    write_submission_zip,
)


class CheckSubmissionPackageTest(unittest.TestCase):
    def author_pattern(self, prefix: str) -> str:
        matches = [pattern for pattern in DOUBLE_BLIND_FORBIDDEN_PATTERNS if pattern.startswith(prefix)]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def write_core_study_artifacts(self, root: Path) -> None:
        for _label, config_relative, summary_relative, expected_seed_count in CORE_STUDY_ARTIFACTS:
            config = root / config_relative
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(
                "experiment:\n  seeds: ["
                + ", ".join(str(seed) for seed in range(expected_seed_count))
                + "]\n",
                encoding="utf-8",
            )

            summary = root / summary_relative
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(
                "method,seed,value\n"
                + "".join(f"bgr,{seed},1.0\n" for seed in range(expected_seed_count)),
                encoding="utf-8",
            )

    def write_generated_table_artifacts(
        self,
        root: Path,
        *,
        corrupt_summary_table: bool = False,
        stale_significance_header: bool = False,
    ) -> None:
        figures = root / "paper" / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        p_column = "two_sided_sign_flip_p" if stale_significance_header else "two_sided_sign_test_p"
        (figures / "significance_tests.csv").write_text(
            "benchmark,condition,metric,treatment,baseline,n,mean_treatment_minus_baseline,paired_se,"
            f"paired_ci95_low,paired_ci95_high,paired_wins,paired_losses,paired_ties,{p_column},direction,supports_treatment\n",
            encoding="utf-8",
        )
        (figures / "bgr_deltas.csv").write_text(
            "benchmark,treatment,baseline,metric,mean_delta,sem_delta\n"
            "Synthetic,BGR,Uniform,RAUC,0.01,0.001\n",
            encoding="utf-8",
        )
        for relative in PAPER_GENERATED_VISUAL_ARTIFACTS:
            (root / relative).write_bytes(b"generated")
        (figures / "boundary_intuition_stats.csv").write_text(
            "metric,value\n"
            "bgr_final_rauc,0.434\n"
            "uniform_final_rauc,0.396\n"
            "failure_only_final_rauc,0.291\n"
            "fixed_final_rauc,0.210\n"
            "plr_loss_final_rauc,0.211\n"
            "bgr_median_r80,0.345\n"
            "uniform_median_r80,0.332\n"
            "failure_only_median_r80,0.300\n"
            "fixed_median_r80,0.200\n"
            "plr_loss_median_r80,0.201\n"
            "uniform_r80_q25,0.303\n"
            "uniform_r80_q75,0.371\n"
            "bgr_r80_q25,0.311\n"
            "bgr_r80_q75,0.398\n"
            "grid_margin_rauc_delta,0.038\n"
            "grid_margin_median_r80_delta,0.013\n"
            "robot_suffix_rauc_delta,0.012\n"
            "robot_suffix_median_r80_delta,-0.007\n",
            encoding="utf-8",
        )

        summary_rows = [
            ("Synthetic", "BGR"),
            ("Synthetic", "Uniform"),
            ("Synthetic", "Fixed"),
            ("Synthetic", "Failure"),
            ("Synthetic", "Loss-priority"),
            ("GridMargin", "BGR"),
            ("GridMargin", "Uniform"),
            ("RobotSuffix", "BGR-Coverage"),
            ("RobotSuffix", "Uniform"),
        ]
        metrics = ["Clean", "RAUC", "MedianR80", "AULC"]
        stats_lines = ["benchmark,method,metric,n,mean,sem\n"]
        table_lines = []
        value = 0.1
        for benchmark, method in summary_rows:
            cells = []
            for metric in metrics:
                stats_lines.append(f"{benchmark},{method},{metric},15,{value:.6f},0.001\n")
                cells.append(f"{value:.3f}$\\pm$0.001")
                value += 0.01
            if corrupt_summary_table and benchmark == "GridMargin" and method == "BGR":
                cells[1] = "9.999$\\pm$0.001"
            table_lines.append(f"{benchmark} & {method} & " + " & ".join(cells) + r" \\")
        (figures / "summary_stats.csv").write_text("".join(stats_lines), encoding="utf-8")
        (figures / "summary_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

        grid_header = "method,n,clean_mean,clean_sem,rauc_mean,rauc_sem,r80_mean,r80_sem,aulc_mean,aulc_sem\n"
        grid_row = "BGR,15,0.946,0.001,0.435,0.001,0.344,0.001,0.353,0.001\n"
        grid_table = r"BGR & 0.946$\pm$0.001 & 0.435$\pm$0.001 & 0.344$\pm$0.001 & 0.353$\pm$0.001 \\"
        for stats_name, table_name in [
            ("grid_margin_full_stats.csv", "grid_margin_full_table.tex"),
            ("grid_margin_ablation_stats.csv", "grid_margin_ablation_table.tex"),
        ]:
            (figures / stats_name).write_text(grid_header + grid_row, encoding="utf-8")
            (figures / table_name).write_text(grid_table, encoding="utf-8")

        (figures / "estimator_stats.csv").write_text(
            "method,n,probes_per_state,r80_mae_mean,r80_mae_sem,rauc_mae_mean,rauc_mae_sem,hit_rate_mean,hit_rate_sem\n"
            "Active BGR,15,17,0.080,0.002,0.065,0.001,0.677,0.007\n",
            encoding="utf-8",
        )
        (figures / "estimator_table.tex").write_text(
            r"Active BGR & 0.080$\pm$0.002 & 0.065$\pm$0.001 & 0.677$\pm$0.007 \\",
            encoding="utf-8",
        )

    def write_environment_artifacts(self, root: Path, *, include_gpu_evidence: bool = True) -> None:
        environment_dir = root / "results" / "environment_v1"
        environment_dir.mkdir(parents=True, exist_ok=True)
        (environment_dir / "README.md").write_text("Environment snapshots.\n", encoding="utf-8")
        (root / "results" / "README.md").write_text(
            "results/environment_v1/compute_environment.json\n"
            "results/environment_v1/gpu_environment.json\n",
            encoding="utf-8",
        )

        base = {
            "commands": {
                "lscpu": {"available": True, "stdout": "CPU"},
                "free_h": {"available": True, "stdout": "Mem"},
            },
            "environment": {
                "SLURM_JOB_ID": "1",
                "SLURM_CPUS_ON_NODE": "2",
                "SLURM_MEM_PER_NODE": "8192",
            },
            "hostname": "node",
            "packages": {"numpy": "2.0.0", "torch": "2.0.0"},
            "platform": {"python_version": "3.10", "system": "Linux", "release": "5.15"},
        }
        compute = dict(base)
        compute["environment"] = dict(base["environment"], SLURM_JOB_PARTITION="compute")
        compute["commands"] = dict(base["commands"], nvidia_proc={"available": False})
        gpu = dict(base)
        gpu["environment"] = dict(
            base["environment"],
            SLURM_JOB_PARTITION="gpu",
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
        )
        gpu["commands"] = dict(
            base["commands"],
            nvidia_proc={"available": include_gpu_evidence, "entries": {"gpu": "NVIDIA RTX A6000"} if include_gpu_evidence else {}},
            lspci_nvidia={"available": include_gpu_evidence, "stdout": "NVIDIA RTX A6000" if include_gpu_evidence else ""},
        )

        (environment_dir / "compute_environment.json").write_text(json.dumps(compute), encoding="utf-8")
        (environment_dir / "gpu_environment.json").write_text(json.dumps(gpu), encoding="utf-8")

    def test_forbidden_terms_detects_old_method_name(self):
        self.assertEqual(forbidden_terms("Boundary-Guided Replay"), [])
        self.assertEqual(forbidden_terms("Bifurcation-Guided Replay"), ["Bifurcation"])
        self.assertEqual(forbidden_terms("paired exact sign-flip tests"), ["sign-flip"])

    def test_bad_log_patterns_detects_latex_failures(self):
        self.assertEqual(bad_log_patterns("Underfull hbox only"), [])
        self.assertEqual(bad_log_patterns("Fatal error\nOverfull box"), ["Overfull", "Fatal"])

    def test_check_log_rejects_stale_caption_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "main.log"
            path.write_text("Package: caption 2020/10/26\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale LaTeX log marker"):
                check_log(path)

    def test_check_log_accepts_underfull_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "main.log"
            path.write_text("Underfull hbox only\n", encoding="utf-8")

            self.assertEqual(check_log(path), [f"{path}: no fatal/rerun/undefined/overfull/stale markers"])

    def test_artifact_doc_framing_accepts_included_artifact_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            readme = Path(temp_dir) / "README.md"
            readme.write_text("The included artifact records configs and result CSVs.\n", encoding="utf-8")

            self.assertEqual(check_artifact_doc_framing(readme), [f"{readme}: artifact doc framing ok"])

    def test_artifact_doc_framing_rejects_positive_checked_in_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            readme = Path(temp_dir) / "README.md"
            readme.write_text("The checked-in artifact records configs and result CSVs.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale artifact doc framing"):
                check_artifact_doc_framing(readme)

    def test_artifact_doc_framing_rejects_checked_into_git_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            readme = Path(temp_dir) / "README.md"
            readme.write_text("Large checkpoint weights are not checked into git.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checked into git"):
                check_artifact_doc_framing(readme)

    def test_artifact_doc_framing_rejects_sign_flip_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            readme = Path(temp_dir) / "README.md"
            readme.write_text("Paired exact sign-flip tests support the result.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sign-flip"):
                check_artifact_doc_framing(readme)

    def test_artifact_doc_framing_rejects_identity_path_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            readme = Path(temp_dir) / "README.md"
            readme.write_text(f"Remote logs are under {self.author_pattern('/work/')}bgr/logs.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "double-blind leak"):
                check_artifact_doc_framing(readme)

    def test_markdown_code_fences_accepts_balanced_fences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            readme = Path(temp_dir) / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "# Reviewer Commands",
                        "",
                        "```bash",
                        "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .",
                        "```",
                        "",
                        "```text",
                        "paper/main.pdf",
                        "```",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_markdown_code_fences(readme),
                [f"{readme}: markdown code fences balanced ok"],
            )

    def test_markdown_code_fences_rejects_unclosed_fence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            readme = Path(temp_dir) / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "# Reviewer Commands",
                        "",
                        "```bash",
                        "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unclosed markdown code fence opened at line 3"):
                check_markdown_code_fences(readme)

    def test_root_readme_command_artifacts_accepts_packaged_script_and_config_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                "PYTHONPATH=src:. python3 scripts/run_toy_experiment.py "
                "--config configs/toy_smoke.yaml --out runs/toy_smoke\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_root_readme_command_artifacts(root),
                [f"{root / 'README.md'}: README command artifact references are packaged ok"],
            )

    def test_root_readme_command_artifacts_rejects_unpackaged_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                "python scripts/not_packaged.py --config configs/missing.yaml\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "not in the anonymous submission manifest"):
                check_root_readme_command_artifacts(root)

    def test_root_readme_command_artifacts_ignores_external_prefixed_script_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                "FINETUNE_SCRIPT=vla-scripts/finetune_identity_lora.py\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_root_readme_command_artifacts(root),
                [f"{root / 'README.md'}: README command artifact references are packaged ok"],
            )

    def test_root_readme_claim_map_artifacts_accepts_packaged_refs_and_anchors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                "\n".join(
                    [
                        "## Claim-Evidence Map",
                        "",
                        "| Paper claim | Primary artifact evidence | Verification hook |",
                        "| --- | --- | --- |",
                        "| Grid claim | `results/grid_margin_full_30seed_v1/summary.csv`, `results/README.md#submission-evidence-index` | `scripts/check_paper_claims.py` |",
                        "",
                        "## Repository Layout",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_root_readme_claim_map_artifacts(root),
                [f"{root / 'README.md'}: Claim-Evidence Map artifact references are packaged ok"],
            )

    def test_root_readme_claim_map_artifacts_rejects_unpackaged_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                "\n".join(
                    [
                        "## Claim-Evidence Map",
                        "",
                        "| Paper claim | Primary artifact evidence | Verification hook |",
                        "| --- | --- | --- |",
                        "| Bad claim | `results/not_packaged/summary.csv` | `scripts/check_submission_package.py` |",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Claim-Evidence Map references artifact"):
                check_root_readme_claim_map_artifacts(root)

    def test_root_readme_artifact_references_accepts_packaged_files_and_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                "\n".join(
                    [
                        "Use `paper/main.pdf` and `results/README.md#submission-evidence-index`.",
                        "Directory references such as `paper/AuthorKit27` and `results/environment_v1` are package scopes.",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_root_readme_artifact_references(root),
                [f"{root / 'README.md'}: README backticked artifact references are packaged ok"],
            )

    def test_root_readme_artifact_references_rejects_unpackaged_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                "Do not point reviewers to `results/not_packaged_v1/summary.csv`.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "README backticked artifact reference"):
                check_root_readme_artifact_references(root)

    def test_required_submission_files_include_complete_bgr_source_package(self):
        files = set(required_submission_files())

        self.assertTrue(set(SOURCE_ARTIFACTS).issubset(files))
        self.assertNotIn("src/bgr/__pycache__/__init__.cpython-314.pyc", files)

    def test_required_submission_files_include_complete_unittest_suite(self):
        files = set(required_submission_files())

        self.assertTrue(set(TEST_ARTIFACTS).issubset(files))
        self.assertIn("tests/test_curve_estimators.py", files)
        self.assertIn("tests/test_suffix_strategy.py", files)

    def test_declared_test_artifacts_match_local_test_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            for relative in TEST_ARTIFACTS:
                path = root / relative
                path.write_text("import unittest\n", encoding="utf-8")

            self.assertEqual(
                check_declared_test_artifacts(root),
                ["all local unittest test files are required submission artifacts ok"],
            )

    def test_declared_test_artifacts_rejects_undeclared_local_test(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tests_dir = root / "tests"
            tests_dir.mkdir()
            for relative in TEST_ARTIFACTS:
                (root / relative).write_text("import unittest\n", encoding="utf-8")
            (tests_dir / "test_new_behavior.py").write_text("import unittest\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing_from_manifest"):
                check_declared_test_artifacts(root)

    def test_license_file_accepts_anonymous_mit_license(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            license_path = Path(temp_dir) / "LICENSE"
            license_path.write_text(
                "MIT License\n\n"
                "Copyright (c) 2026 Anonymous Authors\n\n"
                "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
                "THE SOFTWARE IS PROVIDED \"AS IS\".\n",
                encoding="utf-8",
            )

            self.assertEqual(check_license_file(license_path), [f"{license_path}: anonymous MIT license ok"])

    def test_license_file_rejects_named_author_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            license_path = Path(temp_dir) / "LICENSE"
            author_name = next(pattern for pattern in DOUBLE_BLIND_FORBIDDEN_PATTERNS if " " in pattern and pattern[0].isupper())
            license_path.write_text(
                "MIT License\n\n"
                f"Copyright (c) 2026 {author_name}\n\n"
                "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
                "THE SOFTWARE IS PROVIDED \"AS IS\".\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "double-blind leak"):
                check_license_file(license_path)

    def test_pyproject_runtime_dependencies_cover_reviewer_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pyproject.toml"
            path.write_text(
                '"matplotlib>=3.7"\n"numpy>=1.23"\n"pillow>=9.0"\n"pyyaml>=6.0"\n',
                encoding="utf-8",
            )

            self.assertEqual(
                check_pyproject_runtime_dependencies(path),
                [f"{path}: runtime dependencies cover reviewer commands ok"],
            )

    def test_pyproject_runtime_dependencies_require_matplotlib_for_figure_regeneration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pyproject.toml"
            path.write_text('"numpy>=1.23"\n"pillow>=9.0"\n"pyyaml>=6.0"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "matplotlib"):
                check_pyproject_runtime_dependencies(path)

    def test_double_blind_leaks_detects_soft_repository_identity_markers(self):
        markers = [
            pattern
            for pattern in DOUBLE_BLIND_FORBIDDEN_PATTERNS
            if pattern.startswith("repository is")
            or pattern.endswith("records")
            or pattern.startswith("referenced by")
        ]
        text = f"The {markers[0]}, and raw artifacts are {markers[2]}. The {markers[1]} private logs."

        self.assertEqual(
            double_blind_leaks(text),
            markers,
        )

    def test_core_study_artifacts_require_configs_and_15_seed_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_core_study_artifacts(root)
            (root / CORE_STUDY_ARTIFACTS[0][1]).unlink()

            with self.assertRaisesRegex(ValueError, "missing core study artifact"):
                check_core_study_artifacts(root)

    def test_core_study_artifacts_reject_missing_paired_seed_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_core_study_artifacts(root)
            summary = root / CORE_STUDY_ARTIFACTS[0][2]
            summary.write_text(
                "method,seed,value\n" + "".join(f"bgr,{seed},1.0\n" for seed in range(14)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "result group.*missing configured paired seeds"):
                check_core_study_artifacts(root)

    def test_core_study_artifacts_accept_current_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_core_study_artifacts(root)

            self.assertEqual(
                check_core_study_artifacts(root),
                ["core study configs and paired summary seeds ok"],
            )

    def test_configured_methods_reads_inline_method_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text("experiment:\n  methods: [uniform, bgr_broad]\n", encoding="utf-8")

            self.assertEqual(configured_methods(path), ["uniform", "bgr_broad"])

    def test_suffix_coverage_full_accepts_in_progress_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_coverage_full_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [clean_ft, uniform, fixed, failure_only, loss_priority, bgr_broad]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### In-progress `suffix_coverage_full_30seed_v1`\n"
                "PYTHONPATH=src:. python3 scripts/run_suffix_experiment.py "
                "--config configs/suffix_coverage_full_30seed.yaml "
                "--out results/suffix_coverage_full_30seed_v1\n"
                "Cluster job `763773` is running.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_suffix_coverage_full_status(root),
                [f"{readme}: suffix coverage-full in-progress ledger ok"],
            )

    def test_suffix_coverage_full_rejects_summary_with_in_progress_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_coverage_full_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [clean_ft, uniform, fixed, failure_only, loss_priority, bgr_broad]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### In-progress `suffix_coverage_full_30seed_v1`\n"
                "configs/suffix_coverage_full_30seed.yaml\n"
                "results/suffix_coverage_full_30seed_v1\n"
                "Cluster job `763773` is running.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "suffix_coverage_full_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,final_rauc\n" + "".join(f"bgr_broad,{seed},1.0\n" for seed in range(30)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "ledger is still in-progress"):
                check_suffix_coverage_full_status(root)

    def test_suffix_coverage_full_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_coverage_full_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [clean_ft, uniform, fixed, failure_only, loss_priority, bgr_broad]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `suffix_coverage_full_30seed_v1`\n"
                "Cluster job `763773` completed.\n"
                "BGR-Coverage was evaluated over 30 paired seeds with 30/0 paired wins.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "suffix_coverage_full_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,final_rauc\n" + "".join(f"bgr_broad,{seed},1.0\n" for seed in range(30)),
                encoding="utf-8",
            )

            self.assertEqual(
                check_suffix_coverage_full_status(root),
                [f"{readme}: suffix coverage-full completion ledger ok"],
            )

    def test_suffix_coverage_full_replication_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_coverage_full_replication_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30, 60))
                + "]\n"
                "  methods: [clean_ft, uniform, fixed, failure_only, loss_priority, bgr_broad]\n",
                encoding="utf-8",
            )
            summary = root / "results" / "suffix_coverage_full_replication_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30, 60):
                clean_ft_clean = 0.91 if seed < 52 else 0.89
                rows.extend(
                    [
                        f"clean_ft,{seed},{clean_ft_clean},0.26,0.20,0.18,0.23\n",
                        f"uniform,{seed},0.83,0.48,0.51,0.30,0.37\n",
                        f"fixed,{seed},0.68,0.15,0.12,0.13,0.16\n",
                        f"failure_only,{seed},0.75,0.42,0.47,0.24,0.30\n",
                        f"loss_priority,{seed},0.71,0.16,0.13,0.14,0.17\n",
                        f"bgr_broad,{seed},0.90,0.50,0.49,0.32,0.38\n",
                    ]
                )
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,final_transfer_rauc,rauc_aulc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `suffix_coverage_full_replication_30seed_v1`\n"
                "This held-out run uses seeds 30-59 over 30 paired seeds.\n"
                "BGR-Coverage beats clean-only, fixed-radius, failure-only, loss-priority, and uniform suffix replay on final object RAUC, EE-transfer RAUC, and RAUC AULC with 30/0 paired wins.\n"
                "The clean-only retains a tiny clean-success edge with a 22/8 paired split, and uniform remains higher on median r80 with 30/0 paired wins.\n"
                "Representative means: 0.4972 vs. 0.4859, 0.2634, 0.4237, 0.1590, 0.1699.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_suffix_coverage_full_replication_status(root),
                [f"{readme}: suffix coverage-full replication completion ledger ok"],
            )

    def test_suffix_coverage_full_replication_rejects_weaker_full_baseline_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_coverage_full_replication_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30, 60))
                + "]\n"
                "  methods: [clean_ft, uniform, fixed, failure_only, loss_priority, bgr_broad]\n",
                encoding="utf-8",
            )
            summary = root / "results" / "suffix_coverage_full_replication_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30, 60):
                bgr_rauc = 0.50
                if seed == 30:
                    bgr_rauc = 0.41
                clean_ft_clean = 0.91 if seed < 52 else 0.89
                rows.extend(
                    [
                        f"clean_ft,{seed},{clean_ft_clean},0.26,0.20,0.18,0.23\n",
                        f"uniform,{seed},0.83,0.48,0.51,0.30,0.37\n",
                        f"fixed,{seed},0.68,0.15,0.12,0.13,0.16\n",
                        f"failure_only,{seed},0.75,0.42,0.47,0.24,0.30\n",
                        f"loss_priority,{seed},0.71,0.16,0.13,0.14,0.17\n",
                        f"bgr_broad,{seed},0.90,{bgr_rauc},0.49,0.32,0.38\n",
                    ]
                )
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,final_transfer_rauc,rauc_aulc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `suffix_coverage_full_replication_30seed_v1`\n"
                "seeds 30-59; 30 paired seeds. "
                "BGR-Coverage beats clean-only, fixed-radius, failure-only, loss-priority, and uniform suffix replay on final object RAUC, EE-transfer RAUC, and RAUC AULC with 30/0 paired wins. "
                "clean-only retains a tiny clean-success edge; 22/8 paired split; uniform remains higher on median r80 with 30/0 paired wins. "
                "0.4972 0.4859 0.2634 0.4237 0.1590 0.1699.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 30/0 BGR-Coverage wins over failure_only on final_rauc"):
                check_suffix_coverage_full_replication_status(root)

    def test_suffix_strategy_replication_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_strategy_coverage_replication_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30, 60))
                + "]\n"
                "  methods: [uniform, bgr_broad]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `suffix_strategy_coverage_replication_30seed_v1`\n"
                "This replication uses seeds 30-59 over 30 paired seeds.\n"
                "BGR-Coverage records 30/0 paired wins over uniform, while "
                "uniform remains higher on median r80.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "suffix_strategy_coverage_replication_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,final_transfer_rauc,rauc_aulc\n"
                + "".join(
                    f"uniform,{seed},0.80,0.40,0.60,0.30,0.35\n"
                    f"bgr_broad,{seed},0.90,0.50,0.50,0.40,0.45\n"
                    for seed in range(30, 60)
                ),
                encoding="utf-8",
            )
            results = root / "results" / "suffix_strategy_coverage_replication_30seed_v1" / "results.json"
            results.write_text('{"runs": []}\n', encoding="utf-8")

            self.assertEqual(
                check_suffix_strategy_replication_status(root),
                [f"{readme}: suffix strategy replication completion ledger ok"],
            )

    def test_suffix_strategy_replication_rejects_nonreplicated_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_strategy_coverage_replication_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30, 60))
                + "]\n"
                "  methods: [uniform, bgr_broad]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `suffix_strategy_coverage_replication_30seed_v1`\n"
                "seeds 30-59; 30 paired seeds; 30/0 paired wins; BGR-Coverage; "
                "uniform remains higher on median r80.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "suffix_strategy_coverage_replication_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30, 60):
                uniform_rauc = 0.40
                bgr_rauc = 0.50
                if seed == 30:
                    uniform_rauc = 0.55
                rows.append(f"uniform,{seed},0.80,{uniform_rauc},0.60,0.30,0.35\n")
                rows.append(f"bgr_broad,{seed},0.90,{bgr_rauc},0.50,0.40,0.45\n")
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,final_transfer_rauc,rauc_aulc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            results = root / "results" / "suffix_strategy_coverage_replication_30seed_v1" / "results.json"
            results.write_text('{"runs": []}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected 30/0 BGR-Coverage wins"):
                check_suffix_strategy_replication_status(root)

    def test_suffix_strategy_ablation_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_strategy_ablation_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr_boundary, bgr_broad, bgr_hard]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `suffix_strategy_ablation_30seed_v1`\n"
                "All 120 array tasks completed.\n"
                "BGR-Coverage is the only suffix strategy that improves final object RAUC over uniform with 30/0 paired wins.\n"
                "Boundary-heavy BGR undercovers object RAUC. Hard-radius BGR leads transfer RAUC and RAUC AULC.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "suffix_strategy_ablation_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30):
                rows.extend(
                    [
                        f"uniform,{seed},0.80,0.48,0.50,0.30,0.37\n",
                        f"bgr_boundary,{seed},0.87,0.45,0.43,0.28,0.36\n",
                        f"bgr_broad,{seed},0.86,0.50,0.49,0.31,0.38\n",
                        f"bgr_hard,{seed},0.86,0.485,0.47,0.32,0.39\n",
                    ]
                )
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,final_transfer_rauc,rauc_aulc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            results = root / "results" / "suffix_strategy_ablation_30seed_v1" / "results.json"
            results.write_text('{"results": []}\n', encoding="utf-8")

            self.assertEqual(
                check_suffix_strategy_ablation_status(root),
                [f"{readme}: suffix strategy ablation completion ledger ok"],
            )

    def test_suffix_strategy_ablation_rejects_weaker_coverage_variant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "suffix_strategy_ablation_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr_boundary, bgr_broad, bgr_hard]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `suffix_strategy_ablation_30seed_v1`\n"
                "120 array tasks completed. BGR-Coverage is the only suffix strategy that improves final object RAUC over uniform with 30/0 paired wins. "
                "Boundary-heavy BGR undercovers object RAUC. Hard-radius BGR leads transfer RAUC and RAUC AULC.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "suffix_strategy_ablation_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30):
                hard_rauc = 0.51 if seed == 0 else 0.485
                rows.extend(
                    [
                        f"uniform,{seed},0.80,0.48,0.50,0.30,0.37\n",
                        f"bgr_boundary,{seed},0.87,0.45,0.43,0.28,0.36\n",
                        f"bgr_broad,{seed},0.86,0.50,0.49,0.31,0.38\n",
                        f"bgr_hard,{seed},0.86,{hard_rauc},0.47,0.32,0.39\n",
                    ]
                )
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,final_transfer_rauc,rauc_aulc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            results = root / "results" / "suffix_strategy_ablation_30seed_v1" / "results.json"
            results.write_text('{"results": []}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected 30/0 BGR-Coverage final-RAUC wins over hard-radius"):
                check_suffix_strategy_ablation_status(root)

    def test_grid_margin_full_30_accepts_in_progress_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_full_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, fixed, failure_only, plr_loss, bgr]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### In-progress `grid_margin_full_30seed_v1`\n"
                "PYTHONPATH=src:. python3 scripts/run_grid_margin_experiment.py "
                "--config configs/grid_margin_full_30seed.yaml "
                "--out results/grid_margin_full_30seed_v1\n"
                "Array job `763781` is running.\n"
                "Merge job `763782` will write the summary.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_grid_margin_full_30_status(root),
                [f"{readme}: grid margin full 30-seed in-progress ledger ok"],
            )

    def test_grid_margin_full_30_rejects_summary_with_in_progress_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_full_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, fixed, failure_only, plr_loss, bgr]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### In-progress `grid_margin_full_30seed_v1`\n"
                "configs/grid_margin_full_30seed.yaml\n"
                "results/grid_margin_full_30seed_v1\n"
                "Array job `763781` is running.\n"
                "Merge job `763782` will write the summary.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_full_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,final_rauc\n" + "".join(f"bgr,{seed},1.0\n" for seed in range(30)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "ledger is still in-progress"):
                check_grid_margin_full_30_status(root)

    def test_grid_margin_full_30_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_full_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, fixed, failure_only, plr_loss, bgr]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `grid_margin_full_30seed_v1`\n"
                "BGR was evaluated over 30 paired seeds.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_full_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,final_rauc\n" + "".join(f"bgr,{seed},1.0\n" for seed in range(30)),
                encoding="utf-8",
            )

            self.assertEqual(
                check_grid_margin_full_30_status(root),
                [f"{readme}: grid margin full 30-seed completion ledger ok"],
            )

    def test_toy_30_accepts_completed_ledger_with_rauc_caveat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "toy_bgr_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, fixed, failure_only, plr_loss, bgr]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `toy_30seed_v1`\n"
                "Mean results over 30 paired seeds.\n"
                "BGR improves final RAUC over uniform, 0.3716 vs. 0.3633, with 29/1 paired wins.\n"
                "BGR improves RAUC AULC, 0.3149 vs. 0.2984, with 30/0 paired wins.\n"
                "BGR also beats fixed-radius, failure-only, and PLR-loss baselines on final RAUC.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "toy_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30):
                bgr_rauc = 0.37
                uniform_rauc = 0.36
                if seed == 0:
                    uniform_rauc = 0.38
                rows.append(f"uniform,{seed},0.88,{uniform_rauc},0.28,0.29,0.36\n")
                rows.append(f"fixed,{seed},0.82,0.23,0.13,0.23,0.23\n")
                rows.append(f"failure_only,{seed},0.82,0.23,0.13,0.23,0.23\n")
                rows.append(f"plr_loss,{seed},0.82,0.23,0.13,0.23,0.23\n")
                rows.append(f"bgr,{seed},0.89,{bgr_rauc},0.29,0.31,0.37\n")
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )

            self.assertEqual(
                check_toy_30_status(root),
                [f"{readme}: synthetic 30-seed completion ledger ok"],
            )

    def test_toy_30_rejects_weaker_rauc_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "toy_bgr_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, fixed, failure_only, plr_loss, bgr]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `toy_30seed_v1`\n"
                "30 paired seeds; BGR improves final RAUC over uniform, 0.3716 vs. 0.3633, with 29/1 paired wins.\n"
                "BGR improves RAUC AULC, 0.3149 vs. 0.2984, with 30/0 paired wins.\n"
                "BGR also beats fixed-radius, failure-only, and PLR-loss baselines on final RAUC.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "toy_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30):
                bgr_rauc = 0.37
                uniform_rauc = 0.36
                if seed in {0, 1}:
                    uniform_rauc = 0.38
                rows.append(f"uniform,{seed},0.88,{uniform_rauc},0.28,0.29,0.36\n")
                rows.append(f"fixed,{seed},0.82,0.23,0.13,0.23,0.23\n")
                rows.append(f"failure_only,{seed},0.82,0.23,0.13,0.23,0.23\n")
                rows.append(f"plr_loss,{seed},0.82,0.23,0.13,0.23,0.23\n")
                rows.append(f"bgr,{seed},0.89,{bgr_rauc},0.29,0.31,0.37\n")
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 29/1 BGR wins over uniform on final_rauc"):
                check_toy_30_status(root)

    def test_estimator_30_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "estimator_pair_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, active]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `estimator_pair_30seed_v1`\n"
                "Mean results over 30 paired seeds.\n"
                "The 30-seed confirmation shows active probing improves boundary-hit rate, 0.6701 vs. 0.5949, and lowers mean r80 error, 0.0806 vs. 0.1055, with 30/0 paired wins.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "estimator_pair_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,probes_per_state,r80_mae,r80_bias,rauc_mae,boundary_hit_rate,mean_uncertainty\n"
                + "".join(
                    f"uniform,{seed},17,0.11,0.06,0.066,0.59,0.70\n"
                    f"active,{seed},17,0.08,0.03,0.065,0.67,0.66\n"
                    for seed in range(30)
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_estimator_30_status(root),
                [f"{readme}: estimator 30-seed completion ledger ok"],
            )

    def test_estimator_30_rejects_nonconfirming_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "estimator_pair_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, active]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `estimator_pair_30seed_v1`\n"
                "30 paired seeds; active probing improves boundary-hit rate, 0.6701 vs. 0.5949, and lowers mean r80 error, 0.0806 vs. 0.1055, with 30/0 paired wins.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "estimator_pair_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30):
                active_hit = 0.67
                if seed == 0:
                    active_hit = 0.58
                rows.append(f"uniform,{seed},17,0.11,0.06,0.066,0.59,0.70\n")
                rows.append(f"active,{seed},17,0.08,0.03,0.065,{active_hit},0.66\n")
            summary.write_text(
                "method,seed,probes_per_state,r80_mae,r80_bias,rauc_mae,boundary_hit_rate,mean_uncertainty\n"
                + "".join(rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 30/0 active wins over uniform on boundary_hit_rate"):
                check_estimator_30_status(root)

    def test_grid_margin_ablation_30_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_ablation_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr, bgr_no_uncertainty, bgr_no_sharpness, bgr_uniform_radius]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `grid_margin_ablation_30seed_v1`\n"
                "Mean results over 30 paired seeds.\n"
                "BGR beats the uniform-radius ablation by +0.0525 final RAUC and +0.0475 RAUC AULC with 30/0 paired wins.\n"
                "The uniform-radius ablation is also worse than uniform replay, while no-uncertainty and no-sharpness variants remain effectively tied.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_ablation_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(
                    f"uniform,{seed},0.89,0.40,0.33,0.31,0.40\n"
                    f"bgr,{seed},0.94,0.45,0.34,0.36,0.45\n"
                    f"bgr_no_uncertainty,{seed},0.94,0.45,0.34,0.36,0.45\n"
                    f"bgr_no_sharpness,{seed},0.94,0.45,0.34,0.36,0.45\n"
                    f"bgr_uniform_radius,{seed},0.89,0.38,0.32,0.30,0.38\n"
                    for seed in range(30)
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_grid_margin_ablation_30_status(root),
                [f"{readme}: grid margin ablation 30-seed completion ledger ok"],
            )

    def test_grid_margin_ablation_30_rejects_nonmechanism_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_ablation_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr, bgr_no_uncertainty, bgr_no_sharpness, bgr_uniform_radius]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `grid_margin_ablation_30seed_v1`\n"
                "30 paired seeds; BGR beats the uniform-radius ablation by +0.0525 final RAUC and +0.0475 RAUC AULC with 30/0 paired wins.\n"
                "The uniform-radius ablation is also worse than uniform replay, while no-uncertainty and no-sharpness variants remain effectively tied.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_ablation_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30):
                bgr_rauc = 0.45
                uniform_radius_rauc = 0.38
                if seed == 0:
                    uniform_radius_rauc = 0.46
                rows.append(f"uniform,{seed},0.89,0.40,0.33,0.31,0.40\n")
                rows.append(f"bgr,{seed},0.94,{bgr_rauc},0.34,0.36,0.45\n")
                rows.append(f"bgr_no_uncertainty,{seed},0.94,0.45,0.34,0.36,0.45\n")
                rows.append(f"bgr_no_sharpness,{seed},0.94,0.45,0.34,0.36,0.45\n")
                rows.append(f"bgr_uniform_radius,{seed},0.89,{uniform_radius_rauc},0.32,0.30,0.38\n")
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 30/0 BGR wins over uniform-radius"):
                check_grid_margin_ablation_30_status(root)

    def test_grid_margin_ablation_replication_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_ablation_replication_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30, 60))
                + "]\n"
                "  methods: [uniform, bgr, bgr_no_uncertainty, bgr_no_sharpness, bgr_uniform_radius]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `grid_margin_ablation_replication_30seed_v1`\n"
                "This held-out replication uses held-out seeds 30-59.\n"
                "BGR beats the uniform-radius ablation by +0.0522 final RAUC and +0.0472 RAUC AULC with 30/0 paired wins.\n"
                "The uniform-radius ablation is again worse than uniform replay, while no-uncertainty and no-sharpness variants remain effectively tied.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_ablation_replication_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(
                    f"uniform,{seed},0.89,0.40,0.33,0.31,0.40\n"
                    f"bgr,{seed},0.94,0.45,0.34,0.36,0.45\n"
                    f"bgr_no_uncertainty,{seed},0.94,0.45,0.34,0.36,0.45\n"
                    f"bgr_no_sharpness,{seed},0.94,0.45,0.34,0.36,0.45\n"
                    f"bgr_uniform_radius,{seed},0.89,0.38,0.32,0.30,0.38\n"
                    for seed in range(30, 60)
                ),
                encoding="utf-8",
            )
            results = root / "results" / "grid_margin_ablation_replication_30seed_v1" / "results.json"
            results.write_text('{"results": []}\n', encoding="utf-8")

            self.assertEqual(
                check_grid_margin_ablation_replication_status(root),
                [f"{readme}: grid margin ablation replication completion ledger ok"],
            )

    def test_grid_margin_ablation_replication_rejects_nonmechanism_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_ablation_replication_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30, 60))
                + "]\n"
                "  methods: [uniform, bgr, bgr_no_uncertainty, bgr_no_sharpness, bgr_uniform_radius]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `grid_margin_ablation_replication_30seed_v1`\n"
                "held-out seeds 30-59; BGR beats the uniform-radius ablation by +0.0522 final RAUC and +0.0472 RAUC AULC with 30/0 paired wins.\n"
                "The uniform-radius ablation is again worse than uniform replay, while no-uncertainty and no-sharpness variants remain effectively tied.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_ablation_replication_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30, 60):
                uniform_radius_rauc = 0.38
                if seed == 30:
                    uniform_radius_rauc = 0.46
                rows.append(f"uniform,{seed},0.89,0.40,0.33,0.31,0.40\n")
                rows.append(f"bgr,{seed},0.94,0.45,0.34,0.36,0.45\n")
                rows.append(f"bgr_no_uncertainty,{seed},0.94,0.45,0.34,0.36,0.45\n")
                rows.append(f"bgr_no_sharpness,{seed},0.94,0.45,0.34,0.36,0.45\n")
                rows.append(f"bgr_uniform_radius,{seed},0.89,{uniform_radius_rauc},0.32,0.30,0.38\n")
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            results = root / "results" / "grid_margin_ablation_replication_30seed_v1" / "results.json"
            results.write_text('{"results": []}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected 30/0 held-out BGR wins"):
                check_grid_margin_ablation_replication_status(root)

    def test_grid_margin_target_30_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_target_sensitivity_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  name: grid_margin_target_sensitivity_30seed_v1\n"
                "  base_config: configs/grid_margin_full_30seed.yaml\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  method: bgr\n"
                "  target_margins: [0.26, 0.32, 0.38, 0.46, 0.54]\n",
                encoding="utf-8",
            )
            baseline = root / "results" / "grid_margin_full_30seed_v1" / "summary.csv"
            baseline.parent.mkdir(parents=True)
            baseline.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(f"uniform,{seed},0.89,0.39,0.33,0.31,0.39\n" for seed in range(30)),
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_target_sensitivity_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for margin in ["0.26", "0.32", "0.38", "0.46", "0.54"]:
                for seed in range(30):
                    rows.append(f"bgr,{margin},{seed},0.94,0.42,0.34,0.34,0.42\n")
            summary.write_text(
                "method,target_margin,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `grid_margin_target_sensitivity_30seed_v1`\n"
                "Completed locally over 30 paired seeds.\n"
                "BGR improves final RAUC, RAUC AULC, and clean success over the 30-seed uniform baseline with 30/0 paired wins at every target margin.\n"
                "Mean values include 0.4427, 0.4157, 0.3587, and 0.3393.\n"
                "Interpretation: target 0.38 is not a cherry-picked optimum.\n"
                "Median r80 is not a promoted target-sensitivity claim.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_grid_margin_target_30_status(root),
                [f"{readme}: grid margin target 30-seed completion ledger ok"],
            )

    def test_grid_margin_target_30_rejects_weaker_target_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_target_sensitivity_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  base_config: configs/grid_margin_full_30seed.yaml\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  method: bgr\n"
                "  target_margins: [0.26, 0.32, 0.38, 0.46, 0.54]\n",
                encoding="utf-8",
            )
            baseline = root / "results" / "grid_margin_full_30seed_v1" / "summary.csv"
            baseline.parent.mkdir(parents=True)
            baseline.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(f"uniform,{seed},0.89,0.39,0.33,0.31,0.39\n" for seed in range(30)),
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_target_sensitivity_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for margin in ["0.26", "0.32", "0.38", "0.46", "0.54"]:
                for seed in range(30):
                    final_rauc = 0.42
                    if margin == "0.54" and seed == 0:
                        final_rauc = 0.38
                    rows.append(f"bgr,{margin},{seed},0.94,{final_rauc},0.34,0.34,0.42\n")
            summary.write_text(
                "method,target_margin,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `grid_margin_target_sensitivity_30seed_v1`\n"
                "30 paired seeds; BGR improves final RAUC, RAUC AULC, and clean success over the 30-seed uniform baseline with 30/0 paired wins at every target margin.\n"
                "0.4427 0.4157 0.3587 0.3393 target 0.38 is not a cherry-picked optimum. "
                "Median r80 is not a promoted target-sensitivity claim.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 30/0 BGR target wins over uniform on final_rauc at target_margin=0.54"):
                check_grid_margin_target_30_status(root)

    def test_grid_margin_learning_rate_30_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_learning_rate_sensitivity_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  base_config: configs/grid_margin_full_30seed.yaml\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr]\n"
                "  learning_rates: [0.015, 0.03, 0.06]\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_learning_rate_sensitivity_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for rate in ["0.015", "0.03"]:
                for seed in range(30):
                    rows.append(f"{rate},uniform,{seed},0.88,0.32,0.25,0.27,0.32\n")
                    rows.append(f"{rate},bgr,{seed},0.94,0.38,0.30,0.31,0.38\n")
            for seed in range(30):
                bgr_rauc = 0.48 if seed == 0 else 0.47
                uniform_rauc = 0.47 if seed == 0 else 0.49
                rows.append(f"0.06,uniform,{seed},0.89,{uniform_rauc},0.44,0.37,{uniform_rauc}\n")
                rows.append(f"0.06,bgr,{seed},0.94,{bgr_rauc},0.41,0.39,{bgr_rauc}\n")
            summary.write_text(
                "learning_rate,method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `grid_margin_learning_rate_sensitivity_30seed_v1`\n"
                "Mean results over 30 paired seeds.\n"
                "At low and nominal learning rates, BGR improves final RAUC, RAUC AULC, clean success, and median r80 with 30/0 paired wins.\n"
                "At learning rate 0.060, uniform remains higher on final RAUC with 29/1 paired wins and median r80 with 30/0 paired wins.\n"
                "BGR still improves RAUC AULC and clean success at 0.060 with 30/0 paired wins.\n"
                "This preserves the optimization-scope caveat with 0.3820 vs. 0.3200, 0.4854 vs. 0.4908, and 0.3961 vs. 0.3787.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_grid_margin_learning_rate_30_status(root),
                [f"{readme}: grid margin learning-rate 30-seed completion ledger ok"],
            )

    def test_grid_margin_learning_rate_30_rejects_missing_high_rate_caveat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_learning_rate_sensitivity_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  base_config: configs/grid_margin_full_30seed.yaml\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr]\n"
                "  learning_rates: [0.015, 0.03, 0.06]\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_learning_rate_sensitivity_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for rate in ["0.015", "0.03", "0.06"]:
                for seed in range(30):
                    rows.append(f"{rate},uniform,{seed},0.88,0.32,0.25,0.27,0.32\n")
                    rows.append(f"{rate},bgr,{seed},0.94,0.38,0.30,0.31,0.38\n")
            summary.write_text(
                "learning_rate,method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `grid_margin_learning_rate_sensitivity_30seed_v1`\n"
                "30 paired seeds; low and nominal learning rates; BGR improves final RAUC, RAUC AULC, clean success, and median r80 with 30/0 paired wins. "
                "At learning rate 0.060, uniform remains higher on final RAUC with 29/1 paired wins and median r80 with 30/0 paired wins. "
                "BGR still improves RAUC AULC and clean success at 0.060 with 30/0 paired wins. "
                "optimization-scope caveat 0.3820 0.3200 0.4854 0.4908 0.3961 0.3787.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 1/29 BGR wins over uniform on final_rauc"):
                check_grid_margin_learning_rate_30_status(root)

    def test_grid_margin_regime_30_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_regime_sensitivity_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  base_config: configs/grid_margin_full_30seed.yaml\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr]\n"
                "  regime_names: [low_obstacle, nominal, high_obstacle]\n"
                "  obstacle_probs: [0.14, 0.22, 0.30]\n"
                "  grid_sizes: [11, 11, 11]\n"
                "  max_offsets: [5, 6, 7]\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_regime_sensitivity_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for regime in ["high_obstacle", "low_obstacle", "nominal"]:
                for seed in range(30):
                    rows.append(f"{regime},uniform,0.22,11,6,{seed},0.88,0.32,0.25,0.27,0.32\n")
                    rows.append(f"{regime},bgr,0.22,11,6,{seed},0.94,0.38,0.30,0.31,0.38\n")
            summary.write_text(
                "regime,method,obstacle_prob,grid_size,max_offset,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `grid_margin_regime_sensitivity_30seed_v1`\n"
                "Mean results over 30 paired seeds.\n"
                "BGR improves final RAUC, RAUC AULC, clean success, and median r80 with 30/0 paired wins in every regime.\n"
                "This remains a diagnostic rather than separate robustness evidence because it mostly reproduces the nominal margin dynamics.\n"
                "Representative RAUC means include 0.4346 vs. 0.3963, 0.4345 vs. 0.3963, and 0.4344 vs. 0.3963.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_grid_margin_regime_30_status(root),
                [f"{readme}: grid margin regime 30-seed completion ledger ok"],
            )

    def test_grid_margin_regime_30_rejects_weaker_regime_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_regime_sensitivity_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  base_config: configs/grid_margin_full_30seed.yaml\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr]\n"
                "  regime_names: [low_obstacle, nominal, high_obstacle]\n"
                "  obstacle_probs: [0.14, 0.22, 0.30]\n"
                "  grid_sizes: [11, 11, 11]\n"
                "  max_offsets: [5, 6, 7]\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_regime_sensitivity_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for regime in ["high_obstacle", "low_obstacle", "nominal"]:
                for seed in range(30):
                    bgr_rauc = 0.38
                    if regime == "high_obstacle" and seed == 0:
                        bgr_rauc = 0.31
                    rows.append(f"{regime},uniform,0.22,11,6,{seed},0.88,0.32,0.25,0.27,0.32\n")
                    rows.append(f"{regime},bgr,0.22,11,6,{seed},0.94,{bgr_rauc},0.30,0.31,0.38\n")
            summary.write_text(
                "regime,method,obstacle_prob,grid_size,max_offset,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `grid_margin_regime_sensitivity_30seed_v1`\n"
                "30 paired seeds. BGR improves final RAUC, RAUC AULC, clean success, and median r80 with 30/0 paired wins in every regime. "
                "diagnostic rather than separate robustness evidence; mostly reproduces the nominal margin dynamics. "
                "0.4346 0.3963 0.4345 0.4344.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 30/0 BGR regime wins over uniform on final_rauc"):
                check_grid_margin_regime_30_status(root)

    def test_grid_margin_stress_30_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_stress_sensitivity_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  base_config: configs/grid_margin_full_30seed.yaml\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr]\n"
                '  stress_cases: [{"name": "sharp_low_margin"}, {"name": "diffuse_boundary"}, {"name": "low_feasibility"}]\n',
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_stress_sensitivity_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for case in ["low_feasibility", "sharp_low_margin"]:
                for seed in range(30):
                    rows.append(f"{case},uniform,{seed},0.88,0.32,0.25,0.27,0.32\n")
                    rows.append(f"{case},bgr,{seed},0.94,0.38,0.30,0.31,0.38\n")
            for seed in range(30):
                bgr_median = 0.31 if seed < 14 else 0.29
                rows.append(f"diffuse_boundary,uniform,{seed},0.88,0.43,0.30,0.34,0.43\n")
                rows.append(f"diffuse_boundary,bgr,{seed},0.94,0.46,{bgr_median},0.37,0.46\n")
            summary.write_text(
                "stress_case,method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `grid_margin_stress_sensitivity_30seed_v1`\n"
                "Mean results over 30 paired seeds.\n"
                "BGR improves final RAUC, RAUC AULC, and clean success with 30/0 paired wins in every stress case.\n"
                "The diffuse-boundary median r80 is not promoted because it has a 14/16 paired split.\n"
                "Representative RAUC means: 0.4573 vs. 0.4299, 0.3616 vs. 0.3234, and 0.3980 vs. 0.3250.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_grid_margin_stress_30_status(root),
                [f"{readme}: grid margin stress 30-seed completion ledger ok"],
            )

    def test_grid_margin_stress_30_rejects_weaker_stress_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_stress_sensitivity_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  base_config: configs/grid_margin_full_30seed.yaml\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30))
                + "]\n"
                "  methods: [uniform, bgr]\n"
                '  stress_cases: [{"name": "sharp_low_margin"}, {"name": "diffuse_boundary"}, {"name": "low_feasibility"}]\n',
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_stress_sensitivity_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for case in ["diffuse_boundary", "low_feasibility", "sharp_low_margin"]:
                for seed in range(30):
                    bgr_rauc = 0.38
                    if case == "low_feasibility" and seed == 0:
                        bgr_rauc = 0.31
                    bgr_median = 0.30
                    if case == "diffuse_boundary":
                        bgr_median = 0.31 if seed < 14 else 0.24
                    rows.append(f"{case},uniform,{seed},0.88,0.32,0.25,0.27,0.32\n")
                    rows.append(f"{case},bgr,{seed},0.94,{bgr_rauc},{bgr_median},0.31,0.38\n")
            summary.write_text(
                "stress_case,method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.write_text(
                "### Completed `grid_margin_stress_sensitivity_30seed_v1`\n"
                "30 paired seeds. BGR improves final RAUC, RAUC AULC, and clean success with 30/0 paired wins in every stress case. "
                "diffuse-boundary median r80 is not promoted; 14/16 paired split. "
                "0.4573 0.4299 0.3616 0.3234 0.3980 0.3250.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 30/0 BGR stress wins over uniform on final_rauc"):
                check_grid_margin_stress_30_status(root)

    def test_grid_margin_replication_accepts_completed_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_full_replication_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30, 60))
                + "]\n"
                "  methods: [uniform, bgr]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `grid_margin_full_replication_30seed_v1`\n"
                "This replication uses seeds 30-59 over 30 paired seeds.\n"
                "BGR records 30/0 paired wins over uniform, with 0.4340 vs. 0.3967 final RAUC.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_full_replication_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(
                    f"uniform,{seed},0.80,0.40,0.30,0.35,0.40\n"
                    f"bgr,{seed},0.90,0.50,0.40,0.45,0.50\n"
                    for seed in range(30, 60)
                ),
                encoding="utf-8",
            )
            results = root / "results" / "grid_margin_full_replication_30seed_v1" / "results.json"
            results.write_text('{"results": []}\n', encoding="utf-8")

            self.assertEqual(
                check_grid_margin_replication_status(root),
                [f"{readme}: grid margin replication completion ledger ok"],
            )

    def test_grid_margin_replication_rejects_nonreplicated_signs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "configs" / "grid_margin_full_replication_30seed.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "experiment:\n"
                "  seeds: ["
                + ", ".join(str(seed) for seed in range(30, 60))
                + "]\n"
                "  methods: [uniform, bgr]\n",
                encoding="utf-8",
            )
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "### Completed `grid_margin_full_replication_30seed_v1`\n"
                "seeds 30-59; 30 paired seeds; BGR; 30/0 paired wins; 0.4340 vs. 0.3967.\n",
                encoding="utf-8",
            )
            summary = root / "results" / "grid_margin_full_replication_30seed_v1" / "summary.csv"
            summary.parent.mkdir(parents=True)
            rows = []
            for seed in range(30, 60):
                uniform_rauc = 0.40
                bgr_rauc = 0.50
                if seed == 30:
                    uniform_rauc = 0.55
                rows.append(f"uniform,{seed},0.80,{uniform_rauc},0.30,0.35,0.40\n")
                rows.append(f"bgr,{seed},0.90,{bgr_rauc},0.40,0.45,0.50\n")
            summary.write_text(
                "method,seed,final_clean,final_rauc,final_median_r80,rauc_aulc,best_rauc\n"
                + "".join(rows),
                encoding="utf-8",
            )
            results = root / "results" / "grid_margin_full_replication_30seed_v1" / "results.json"
            results.write_text('{"results": []}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected 30/0 BGR wins"):
                check_grid_margin_replication_status(root)

    def write_results_evidence_index(self, readme: Path) -> None:
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(
            "\n".join(
                [
                    "# Experiment Results Ledger",
                    "",
                    "## Submission Evidence Index",
                    "",
                    "Primary controlled evidence for the paper is:",
                    "",
                    "- `results/toy_30seed_v1/summary.csv`: 30 paired seeds for the controlled synthetic recovery-margin benchmark.",
                    "- `results/toy_15seed_v1/summary.csv`: original 15 paired-seed synthetic mechanism run retained for provenance.",
                    "- `results/grid_margin_full_30seed_v1/summary.csv`: 30 paired seeds.",
                    "- `results/grid_margin_full_replication_30seed_v1/summary.csv`: held-out replication.",
                    "- `results/suffix_coverage_full_30seed_v1/summary.csv`: 30 paired seeds for the coverage-aware robot-suffix full-baseline comparison against clean-only, fixed-radius, failure-only, loss-priority, and uniform replay.",
                    "- `results/suffix_coverage_full_replication_30seed_v1/summary.csv`: held-out suffix full-baseline replication.",
                    "- `results/suffix_strategy_coverage_30seed_v1/summary.csv`: 30 paired seeds.",
                    "- `results/suffix_strategy_coverage_replication_30seed_v1/summary.csv`: held-out replication.",
                    "- `results/suffix_strategy_ablation_30seed_v1/summary.csv`: 30-seed suffix strategy ablation.",
                    "- `results/suffix_stress_sensitivity_30seed_v1/summary.csv`: 30-seed suffix stress sweep over low-teacher-quality, high-clutter, tight-feasibility, and diffuse-boundary cases.",
                    "- `paper/figures/suffix_stress_sensitivity_stats.csv`: generated suffix stress table.",
                    "- `paper/figures/significance_tests.csv`: paired exact sign tests.",
                    "- `paper/figures/estimator_stats.csv` and `paper/figures/estimator_table.tex`:",
                    "  method-validation evidence that active boundary probing estimates useful critical radii at a small fixed rollout budget.",
                    "- `results/estimator_pair_30seed_v1/summary.csv`: 30-seed active-estimator confirmation.",
                    "Packaged grid-margin robustness/scope diagnostics are:",
                    "- `results/grid_margin_ablation_30seed_v1/summary.csv`: 30-seed radius-level ablation showing that boundary-radius sampling is the important BGR component.",
                    "- `results/grid_margin_ablation_replication_30seed_v1/summary.csv`: held-out seeds 30--59 replication of the radius-level ablation.",
                    "- `results/grid_margin_ablation_15seed_v1/summary.csv`: original 15-seed radius-level ablation retained for provenance.",
                    "- `paper/figures/grid_margin_learning_curve_stats.csv` and `results/grid_margin_full_15seed_v1/results.json`: stored 15-seed learning-curve history and generated stats.",
                    "- `paper/figures/grid_margin_target_sensitivity_stats.csv` and `results/grid_margin_target_sensitivity_30seed_v1/summary.csv`: 30-seed target-margin table/source; `results/grid_margin_target_sensitivity_15seed_v1/summary.csv` is retained for provenance.",
                    "- `paper/figures/grid_margin_learning_rate_sensitivity_stats.csv` and `results/grid_margin_learning_rate_sensitivity_30seed_v1/summary.csv`: 30-seed learning-rate table/source; `results/grid_margin_learning_rate_sensitivity_15seed_v1/summary.csv` is retained for provenance.",
                    "- `paper/figures/grid_margin_regime_sensitivity_stats.csv` and `results/grid_margin_regime_sensitivity_30seed_v1/summary.csv`: 30-seed regime table/source; `results/grid_margin_regime_sensitivity_15seed_v1/summary.csv` is retained for provenance.",
                    "- `paper/figures/grid_margin_stress_sensitivity_stats.csv` and `results/grid_margin_stress_sensitivity_30seed_v1/summary.csv`: 30-seed geometry-stress table/source; `results/grid_margin_stress_sensitivity_15seed_v1/summary.csv` is retained for provenance.",
                    "- `paper/figures/suffix_stress_sensitivity_stats.csv` and `results/suffix_stress_sensitivity_30seed_v1/summary.csv`: 30-seed suffix stress table/source; `results/suffix_stress_sensitivity_15seed_v1/summary.csv` is retained for provenance.",
                    "",
                    "Secondary diagnostics are included to scope the claim rather than expand it.",
                    "OpenVLA/LIBERO rows are recovery-curve, selection, and data-plumbing audits.",
                    "Packaged OpenVLA audit artifacts are:",
                    "- `results/libero_probe_v2/summary.csv`: resettable LIBERO radius-probe audit.",
                    "- `paper/figures/openvla_stats.csv`: OpenVLA recovery and selection audit stats.",
                    "- `results/libero_openvla_recovery_v1/summary.csv`: source recovery summary used to generate the OpenVLA audit stats.",
                    "- `results/libero_openvla_boundary_selection_balanced_v1/aggregate.csv`: source selection summary used to generate the OpenVLA audit stats.",
                    "- `results/openvla_teacher_replay_manifest_v1/summary.json`: teacher-replay data-plumbing audit.",
                    "- `results/openvla_action_tfds_validation_v1/summary.json`: compact action-label/TFDS plumbing audit validating 2,048-transition matched BGR/random exports with 7D actions, 8D state, stock loader ingestion, and matched 10-step checkpoint smokes.",
                    "- `results/openvla_oft_sanity_eval_sanity_v1/summary.csv`: official-checkpoint sanity audit.",
                    "- `results/openvla_oft_eval_balanced2048_step1000_v1/summary.csv`: 1,000-step balanced2048 data-plumbing audit.",
                    "- `results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`: p1024 clean adaptation audit.",
                    "- `results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`: p1024 original perturbation audit.",
                    "- `results/openvla_oft_perturb_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv`: p1024 offset-3 perturbation audit.",
                    "- `results/openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`: p2048 clean adaptation audit.",
                    "- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`: p2048 original perturbation audit.",
                    "- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv`: p2048 offset-3 perturbation audit.",
                    "- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_10trials_v1/summary.csv`: p2048 10-trial perturbation variance audit.",
                    "- `results/openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv`: p2048 full-goal clean identity audit.",
                    "- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv`: p2048 full-goal visual perturbation audit.",
                    "- `results/openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1/summary.csv`: p2048 300-step image-augmentation continuation audit.",
                    "- `results/openvla_oft_perturb_eval_cleanmix_p2048_step51000_lr1em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1/summary.csv`: p2048 1,000-step low-LR image-augmentation continuation audit.",
                    "Packaged FrozenLake diagnostic:",
                    "- `results/frozenlake_recovery_focused_30seed_v1/summary.csv`: canonical Gym FrozenLake8x8-v1 limitation diagnostic.",
                    "- `results/reacher_recovery_probe_12seed_v1/summary.csv`: fixed Reacher-v5 limitation diagnostic.",
                    "The p4096 and common-availability sections below are retained as paper-negative diagnostics in this ledger only.",
                    "Their summary CSVs are not part of the anonymous submission manifest or archive.",
                    "",
                    "The detailed sections below are a historical provenance ledger.",
                    "Reviewer-facing evidence should be read from the index above and the anonymous manuscript;",
                    "later diagnostic sections are included for auditability.",
                    "Older troubleshooting sections may retain labels such as Queued command to record original Slurm submissions;",
                    "those labels are provenance, not active experiment status.",
                    "",
                    "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                ]
            ),
            encoding="utf-8",
        )

    def write_root_claim_map_with_checked_artifacts(self, readme: Path, missing: set[str] | None = None) -> None:
        missing = missing or set()
        readme.write_text(
            "\n".join(
                [
                    "# Boundary-Guided Replay",
                    "",
                    "## Claim-Evidence Map",
                    "",
                    *[
                        f"- `{relative}`: checked paper-claim evidence."
                        for relative in CHECKED_CLAIM_ARTIFACTS
                        if relative not in missing
                    ],
                    "",
                    "## Repository Layout",
                ]
            ),
            encoding="utf-8",
        )

    def test_results_evidence_index_accepts_positive_leading_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)

            self.assertEqual(
                check_results_evidence_index(root),
                [f"{readme}: submission evidence index leads paper-negative diagnostics ok"],
            )

    def test_results_evidence_index_rejects_missing_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text("## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing submission evidence index"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_p4096_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic\n" + readme.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must precede p4096"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_stale_current_p1024_superlative(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nInterpretation: p1024 is the strongest current OpenVLA-OFT diagnostic.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_bounded_openvla_finetune_bridge_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nInterpretation: both matched 2,048-step TFDS exports now complete bounded "
                "OpenVLA-OFT LoRA fine-tuning runs.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_recent_openvla_heading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8") + "\n## Recent OpenVLA-OFT Pilot\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_forward_looking_openvla_ledger_framing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nThe next OpenVLA experiment should evaluate the corrected checkpoints.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_superseded_openvla_bridge_gap_framing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nThis is the next bridge artifact toward RLDS conversion, "
                "not a completed RLDS dataset or fine-tuning run.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_openvla_troubleshooting_maintenance_framing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nThe launcher now defaults to SERIAL_TRAIN=1. "
                "The queue script now skips git pull by default.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_live_notebook_result_framing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nThe result narrows the remaining robotics gap to data scale.\n"
                + "Queued on 2026-06-03 to confirm the procedural result.\n"
                + "The suffix simulator is now positive on final object RAUC.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_unscoped_packaged_openvla_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            text = readme.read_text(encoding="utf-8").replace(
                "p1024 clean adaptation audit",
                "p1024 clean adaptation diagnostic",
            )
            readme.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale packaged OpenVLA evidence-index label"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_bounded_scope_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            text = readme.read_text(encoding="utf-8").replace(
                "Secondary diagnostics are included to scope the claim rather than expand it.",
                "Secondary diagnostics are included to bound the claim rather than expand it.",
            )
            readme.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale evidence-index scope label"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_paper_priority_ordering_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            text = readme.read_text(encoding="utf-8").replace(
                "The detailed sections below are a historical provenance ledger.",
                "The detailed sections below are a historical run ledger, not a paper-priority ordering.",
            )
            readme.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale evidence-index scope label"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_stale_suffix_full_spec_framing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nTreat this as a robotics-style diagnostic rather than the final LIBERO/OpenVLA evidence requested by the full spec.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_stale_grid_margin_v1_main_paper_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nThe main paper should continue using `grid_margin_full_v1` as the positive procedural result.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_stale_active_grid_next_step_guidance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            readme.write_text(
                readme.read_text(encoding="utf-8")
                + "\nDo not use this as a positive main-paper result. The next iteration should either harden perturbations/generalization.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale current OpenVLA ledger framing"):
                check_results_evidence_index(root)

    def test_results_evidence_index_rejects_unpackaged_artifact_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            text = readme.read_text(encoding="utf-8")
            text = text.replace(
                "- `paper/figures/significance_tests.csv`: paired exact sign tests.",
                "- `paper/figures/significance_tests.csv`: paired exact sign tests.\n"
                "- `results/not_packaged_v1/summary.csv`: should not be listed.",
            )
            readme.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Submission Evidence Index references artifact"):
                check_results_evidence_index(root)

    def test_results_evidence_index_requires_ledger_order_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "results" / "README.md"
            self.write_results_evidence_index(readme)
            text = readme.read_text(encoding="utf-8")
            text = text.replace(
                "\nThe detailed sections below are a historical provenance ledger.\n"
                "Reviewer-facing evidence should be read from the index above and the anonymous manuscript;\n"
                "later diagnostic sections are included for auditability.\n",
                "\n",
            )
            readme.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "historical provenance ledger"):
                check_results_evidence_index(root)

    def test_checked_claim_artifact_evidence_maps_accept_complete_maps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_root_claim_map_with_checked_artifacts(root / "README.md")
            self.write_results_evidence_index(root / "results" / "README.md")

            self.assertEqual(
                check_checked_claim_artifact_evidence_maps(root),
                ["checked claim artifacts are exposed in reviewer evidence maps ok"],
            )

    def test_checked_claim_artifact_evidence_maps_require_root_claim_map_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = "paper/figures/significance_tests.csv"
            self.write_root_claim_map_with_checked_artifacts(root / "README.md", missing={missing})
            self.write_results_evidence_index(root / "results" / "README.md")

            with self.assertRaisesRegex(ValueError, f"README.md: {missing}"):
                check_checked_claim_artifact_evidence_maps(root)

    def test_checked_claim_artifact_evidence_maps_require_results_index_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = "results/grid_margin_ablation_15seed_v1/summary.csv"
            self.write_root_claim_map_with_checked_artifacts(root / "README.md")
            results_readme = root / "results" / "README.md"
            self.write_results_evidence_index(results_readme)
            results_readme.write_text(
                results_readme.read_text(encoding="utf-8").replace(
                    f"- `{missing}`: original 15-seed radius-level ablation retained for provenance.\n",
                    "",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, f"results/README.md: {missing}"):
                check_checked_claim_artifact_evidence_maps(root)

    def test_generated_result_tables_reject_stale_rounded_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_generated_table_artifacts(root, corrupt_summary_table=True)

            with self.assertRaisesRegex(ValueError, "missing generated table row"):
                check_generated_result_tables(root)

    def test_generated_result_tables_reject_stale_significance_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_generated_table_artifacts(root, stale_significance_header=True)

            with self.assertRaisesRegex(ValueError, "stale sign-flip terminology"):
                check_generated_result_tables(root)

    def test_generated_result_tables_require_visual_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_generated_table_artifacts(root)
            (root / PAPER_GENERATED_VISUAL_ARTIFACTS[0]).unlink()

            with self.assertRaisesRegex(ValueError, "missing generated visual artifact"):
                check_generated_result_tables(root)

    def test_generated_result_tables_require_boundary_r80_distribution_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_generated_table_artifacts(root)
            stats_path = root / "paper" / "figures" / "boundary_intuition_stats.csv"
            stats_text = stats_path.read_text(encoding="utf-8")
            stats_path.write_text(stats_text.replace("bgr_r80_q75,0.398\n", ""), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "bgr_r80_q75"):
                check_generated_result_tables(root)

    def test_generated_result_tables_reject_incoherent_boundary_r80_distribution_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_generated_table_artifacts(root)
            stats_path = root / "paper" / "figures" / "boundary_intuition_stats.csv"
            stats_text = stats_path.read_text(encoding="utf-8")
            stats_path.write_text(stats_text.replace("bgr_r80_q75,0.398", "bgr_r80_q75,0.320"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "critical-radius distribution quantiles"):
                check_generated_result_tables(root)

    def test_generated_result_tables_accept_stats_synced_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_generated_table_artifacts(root)

            self.assertEqual(check_generated_result_tables(root), ["generated result tables match figure stats ok"])

    def test_aggregate_text_artifacts_accept_exact_regeneration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated = root / "generated"
            expected = root / "paper" / "figures"
            generated.mkdir(parents=True)
            expected.mkdir(parents=True)
            for filename in AGGREGATE_TEXT_ARTIFACTS:
                text = f"{filename}\n"
                (generated / filename).write_text(text, encoding="utf-8")
                (expected / filename).write_text(text, encoding="utf-8")

            self.assertEqual(
                check_aggregate_text_artifacts_synced(root, generated),
                ["aggregate result CSV/TEX artifacts regenerate exactly ok"],
            )

    def test_aggregate_text_artifacts_reject_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated = root / "generated"
            expected = root / "paper" / "figures"
            generated.mkdir(parents=True)
            expected.mkdir(parents=True)
            for filename in AGGREGATE_TEXT_ARTIFACTS:
                (generated / filename).write_text(f"{filename}\n", encoding="utf-8")
                (expected / filename).write_text(f"{filename}\n", encoding="utf-8")
            (expected / AGGREGATE_TEXT_ARTIFACTS[0]).write_text("stale\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "drift from aggregate_results.py"):
                check_aggregate_text_artifacts_synced(root, generated)

    def test_aggregate_outputs_synced_uses_writable_matplotlib_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = root / "paper" / "figures"
            expected.mkdir(parents=True)
            for filename in AGGREGATE_TEXT_ARTIFACTS:
                (expected / filename).write_text(f"{filename}\n", encoding="utf-8")
            seen_env = {}

            def fake_run(command, **kwargs):
                out_dir = Path(command[-1])
                for filename in AGGREGATE_TEXT_ARTIFACTS:
                    (out_dir / filename).write_text(f"{filename}\n", encoding="utf-8")
                seen_env.update(kwargs["env"])
                mplconfig = Path(kwargs["env"]["MPLCONFIGDIR"])
                self.assertTrue(mplconfig.exists())
                self.assertTrue(mplconfig.is_dir())
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("scripts.check_submission_package.subprocess.run", side_effect=fake_run):
                self.assertEqual(
                    check_aggregate_outputs_synced(root),
                    ["aggregate result CSV/TEX artifacts regenerate exactly ok"],
                )

            self.assertIn("MPLCONFIGDIR", seen_env)

    def test_significance_text_artifacts_accept_exact_regeneration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated = root / "generated"
            expected = root / "paper" / "figures"
            generated.mkdir(parents=True)
            expected.mkdir(parents=True)
            for filename in SIGNIFICANCE_TEXT_ARTIFACTS:
                text = f"{filename}\n"
                (generated / filename).write_text(text, encoding="utf-8")
                (expected / filename).write_text(text, encoding="utf-8")

            self.assertEqual(
                check_significance_text_artifacts_synced(root, generated),
                ["significance CSV/TEX artifacts regenerate exactly ok"],
            )

    def test_toy_smoke_experiment_runs_documented_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir()
            (root / "scripts").mkdir()
            (root / "configs" / "toy_smoke.yaml").write_text("experiment:\n", encoding="utf-8")
            (root / "scripts" / "run_toy_experiment.py").write_text("# smoke\n", encoding="utf-8")
            seen_env = {}

            def fake_run(command, **kwargs):
                out_dir = Path(command[-1])
                out_dir.mkdir(parents=True)
                (out_dir / "summary.csv").write_text(
                    "method,seed,final_rauc\nuniform,0,0.1\nbgr,0,0.2\n",
                    encoding="utf-8",
                )
                (out_dir / "results.json").write_text('{"results": [{}, {}]}\n', encoding="utf-8")
                (out_dir / "rauc_learning_curve.png").write_bytes(b"png")
                seen_env.update(kwargs["env"])
                self.assertEqual(command[1], "scripts/run_toy_experiment.py")
                self.assertIn("configs/toy_smoke.yaml", command)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("scripts.check_submission_package.subprocess.run", side_effect=fake_run):
                self.assertEqual(
                    check_toy_smoke_experiment(root),
                    ["toy smoke experiment command runs and writes expected outputs ok"],
                )

            self.assertIn(str(root / "src"), seen_env["PYTHONPATH"])

    def test_toy_smoke_experiment_rejects_missing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir()
            (root / "scripts").mkdir()
            (root / "configs" / "toy_smoke.yaml").write_text("experiment:\n", encoding="utf-8")
            (root / "scripts" / "run_toy_experiment.py").write_text("# smoke\n", encoding="utf-8")

            def fake_run(command, **kwargs):
                Path(command[-1]).mkdir(parents=True)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("scripts.check_submission_package.subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(ValueError, "missed output"):
                    check_toy_smoke_experiment(root)

    def test_significance_text_artifacts_reject_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated = root / "generated"
            expected = root / "paper" / "figures"
            generated.mkdir(parents=True)
            expected.mkdir(parents=True)
            for filename in SIGNIFICANCE_TEXT_ARTIFACTS:
                (generated / filename).write_text(f"{filename}\n", encoding="utf-8")
                (expected / filename).write_text(f"{filename}\n", encoding="utf-8")
            (expected / SIGNIFICANCE_TEXT_ARTIFACTS[0]).write_text("stale\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "drift from analyze_significance.py"):
                check_significance_text_artifacts_synced(root, generated)

    def test_environment_artifacts_accept_compute_and_gpu_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_environment_artifacts(root)

            self.assertEqual(check_environment_artifacts(root), ["compute/GPU environment snapshots ok"])

    def test_environment_artifacts_require_both_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_environment_artifacts(root)
            (root / "results" / "environment_v1" / "gpu_environment.json").unlink()

            with self.assertRaisesRegex(ValueError, "missing environment artifact"):
                check_environment_artifacts(root)

    def test_environment_artifacts_reject_missing_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_environment_artifacts(root, include_gpu_evidence=False)

            with self.assertRaisesRegex(ValueError, "missing NVIDIA GPU evidence"):
                check_environment_artifacts(root)

    def test_environment_artifacts_reject_identity_path_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_environment_artifacts(root)
            gpu_path = root / "results" / "environment_v1" / "gpu_environment.json"
            gpu_text = gpu_path.read_text(encoding="utf-8")
            gpu_path.write_text(
                gpu_text.replace(
                    '"python_version": "3.10"',
                    f'"python_version": "3.10", "python_executable": "{self.author_pattern("/users/")}env/bin/python"',
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "double-blind leak"):
                check_environment_artifacts(root)

    def test_openvla_bridge_scripts_and_tests_are_required_submission_artifacts(self):
        required = set(required_submission_files())

        self.assertTrue(set(OPENVLA_BRIDGE_ARTIFACTS).issubset(required))
        self.assertTrue(set(OPENVLA_BRIDGE_TEST_ARTIFACTS).issubset(required))

    def test_generated_visuals_are_required_submission_artifacts(self):
        required = set(required_submission_files())

        self.assertTrue(set(PAPER_GENERATED_VISUAL_ARTIFACTS).issubset(required))
        self.assertIn("paper/figures/bgr_deltas.csv", required)

    def test_data_artifact_text_files_are_required_submission_artifacts(self):
        required = set(required_submission_files())

        self.assertTrue(set(data_artifact_text_files()).issubset(required))

    def test_aggregate_source_artifacts_are_required_submission_artifacts(self):
        required = set(required_submission_files())

        self.assertTrue(set(AGGREGATE_SOURCE_ARTIFACTS).issubset(required))

    def test_significance_source_artifacts_are_required_submission_artifacts(self):
        required = set(required_submission_files())

        self.assertTrue(set(SIGNIFICANCE_SOURCE_ARTIFACTS).issubset(required))

    def test_author_kit_files_are_required_submission_artifacts(self):
        required = set(required_submission_files())

        self.assertTrue(set(AUTHOR_KIT_ARTIFACTS).issubset(required))

    def test_submission_manifest_is_required_submission_artifact(self):
        self.assertIn(SUBMISSION_MANIFEST, required_submission_files())

    def test_tex_logs_are_required_submission_artifacts(self):
        required = set(required_submission_files())

        self.assertTrue(ALLOWED_TEX_LOG_ARTIFACTS.issubset(required))

    def test_required_artifact_scope_excludes_current_raw_run_outputs(self):
        self.assertEqual(
            check_required_artifact_scope(Path(".")),
            ["required submission artifact scope excludes raw logs, container workflows, and paper-negative diagnostics ok"],
        )

    def test_required_artifact_scope_excludes_current_paper_negative_outputs(self):
        required = required_submission_files()

        self.assertFalse(any("p4096" in relative.lower() for relative in required))
        self.assertFalse(any("commonavail" in relative.lower() for relative in required))

    def test_required_artifact_scope_rejects_raw_slurm_log_paths(self):
        required = required_submission_files()
        with mock.patch(
            "scripts.check_submission_package.required_submission_files",
            return_value=required
            + [
                "results/openvla_oft_eval/slurm/job-1.out",
                "results/openvla_oft_eval/logs/job.err",
                "runs/grid/slurm/job.log",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "raw run/log output"):
                check_required_artifact_scope(Path("."))

    def test_container_workflow_artifact_paths_rejects_docker_workflows(self):
        self.assertEqual(
            container_workflow_artifact_paths(
                [
                    "Dockerfile",
                    "docker-compose.yml",
                    ".devcontainer/devcontainer.json",
                    "scripts/queue_openvla_oft_eval.sh",
                ]
            ),
            [".devcontainer/devcontainer.json", "Dockerfile", "docker-compose.yml"],
        )

    def test_required_artifact_scope_rejects_container_workflows(self):
        required = required_submission_files()
        with mock.patch(
            "scripts.check_submission_package.required_submission_files",
            return_value=required + ["Dockerfile", ".devcontainer/devcontainer.json"],
        ):
            with self.assertRaisesRegex(ValueError, "container workflow"):
                check_required_artifact_scope(Path("."))

    def test_required_artifact_scope_rejects_paper_negative_diagnostics(self):
        required = required_submission_files()
        with mock.patch(
            "scripts.check_submission_package.required_submission_files",
            return_value=required
            + [
                "results/openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv",
                "results/openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "paper-negative diagnostic output"):
                check_required_artifact_scope(Path("."))

    def test_tex_dependencies_are_required_submission_artifacts(self):
        dependencies = set(tex_source_dependency_files(Path(".")))
        required = set(required_submission_files())

        self.assertTrue(dependencies)
        self.assertTrue(dependencies.issubset(required))

    def test_data_artifact_double_blind_accepts_anonymous_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in data_artifact_text_files():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact_path,/work/anonymous/bgr/results\n", encoding="utf-8")

            self.assertEqual(
                check_data_artifact_double_blind(root),
                ["data/result/figure text artifacts are double-blind safe ok"],
            )

    def test_data_artifact_double_blind_rejects_identity_path_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in data_artifact_text_files():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("artifact_path,/work/anonymous/bgr/results\n", encoding="utf-8")
            leaked = root / data_artifact_text_files()[0]
            leaked.write_text(f"artifact_path,{self.author_pattern('/work/')}bgr/results\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "data/result/figure text artifact double-blind leak"):
                check_data_artifact_double_blind(root)

    def test_required_text_artifacts_include_required_source_and_exclude_binaries(self):
        text_artifacts = set(required_text_artifact_files())

        self.assertIn(SUBMISSION_MANIFEST, text_artifacts)
        self.assertTrue(set(AUTHOR_KIT_ARTIFACTS).issubset(text_artifacts))
        self.assertIn("scripts/check_submission_package.py", text_artifacts)
        self.assertIn("tests/test_check_submission_package.py", text_artifacts)
        self.assertNotIn("paper/main.pdf", text_artifacts)
        self.assertNotIn("paper/figures/aulc_bars.png", text_artifacts)

    def test_required_text_artifacts_double_blind_accepts_anonymous_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in required_text_artifact_files():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("anonymous artifact text\n", encoding="utf-8")

            self.assertEqual(
                check_required_text_artifacts_double_blind(root),
                ["required text artifacts are double-blind safe ok"],
            )

    def test_required_text_artifacts_double_blind_rejects_identity_path_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in required_text_artifact_files():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("anonymous artifact text\n", encoding="utf-8")
            leaked = root / "scripts" / "aggregate_results.py"
            leaked.write_text(f"REMOTE_PROJECT = '{self.author_pattern('/work/')}bgr'\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "required text artifact double-blind leak"):
                check_required_text_artifacts_double_blind(root)

    def test_submission_manifest_accepts_current_required_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in required_submission_files():
                if relative == SUBMISSION_MANIFEST:
                    continue
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{relative}\n", encoding="utf-8")
            manifest = root / SUBMISSION_MANIFEST
            manifest.write_text(canonical_json(submission_manifest_payload(root)), encoding="utf-8")

            self.assertEqual(
                check_submission_manifest(root),
                [f"{SUBMISSION_MANIFEST}: required artifact SHA-256 manifest ok"],
            )

    def test_submission_manifest_rejects_stale_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in required_submission_files():
                if relative == SUBMISSION_MANIFEST:
                    continue
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{relative}\n", encoding="utf-8")
            manifest = root / SUBMISSION_MANIFEST
            manifest.write_text(canonical_json(submission_manifest_payload(root)), encoding="utf-8")
            changed = root / "README.md"
            changed.write_text("changed\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "required artifact hashes are stale"):
                check_submission_manifest(root)

    def test_submission_manifest_docs_accepts_verification_and_regeneration_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "submission_manifest.json records the SHA-256 manifest.\n"
                "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .\n"
                "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . --write-required-manifest\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_submission_manifest_docs(root),
                [f"{readme}: submission manifest verification docs ok"],
            )

    def test_submission_manifest_docs_requires_regeneration_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "submission_manifest.json records the SHA-256 manifest.\n"
                "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "write-required-manifest"):
                check_submission_manifest_docs(root)

    def test_submission_scope_docs_accepts_manifest_only_artifact_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                (
                    "The anonymous submission archive contains `submission_manifest.json` plus the "
                    "files it declares.\n"
                    "Only those archive entries should be treated as the anonymous submission artifact.\n"
                    "The manifest hashes the declared payload files; the manifest itself is packaged "
                    "as the verifier entry point and is intentionally not self-hashed.\n"
                    "Mirrored raw Slurm logs are diagnostic run records and are not part of the "
                    "anonymous submission artifact.\n"
                    "Cluster commands below are provenance recipes; any remote input paths they "
                    "mention are not reviewer evidence unless their generated summaries are declared "
                    "in `submission_manifest.json`.\n"
                    "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . "
                    "--write-submission-zip bgr-aaai27-anonymous.zip\n"
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_submission_scope_docs(root),
                [f"{readme}: anonymous submission scope docs ok"],
            )

    def test_submission_scope_docs_requires_raw_log_exclusion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                (
                    "The anonymous submission archive contains `submission_manifest.json` plus the "
                    "files it declares.\n"
                    "Only those archive entries should be treated as the anonymous submission artifact.\n"
                    "The manifest hashes the declared payload files; the manifest itself is packaged "
                    "as the verifier entry point and is intentionally not self-hashed.\n"
                    "Mirrored raw Slurm logs are diagnostic run records and are not part of the "
                    "anonymous submission artifact.\n"
                    "Cluster commands below are provenance recipes; any remote input paths they "
                    "mention are not reviewer evidence unless their generated summaries are declared "
                    "in `submission_manifest.json`.\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "write-submission-zip"):
                check_submission_scope_docs(root)

    def test_submission_scope_docs_requires_manifest_self_hash_clarification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                (
                    "The anonymous submission archive contains `submission_manifest.json` plus the "
                    "files it declares.\n"
                    "Only those archive entries should be treated as the anonymous submission artifact.\n"
                    "Mirrored raw Slurm logs are diagnostic run records and are not part of the "
                    "anonymous submission artifact.\n"
                    "Cluster commands below are provenance recipes; any remote input paths they "
                    "mention are not reviewer evidence unless their generated summaries are declared "
                    "in `submission_manifest.json`.\n"
                    "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . "
                    "--write-submission-zip bgr-aaai27-anonymous.zip\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "not self-hashed"):
                check_submission_scope_docs(root)

    def test_submission_scope_docs_requires_remote_path_provenance_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                (
                    "The anonymous submission archive contains `submission_manifest.json` plus the "
                    "files it declares.\n"
                    "Only those archive entries should be treated as the anonymous submission artifact.\n"
                    "The manifest hashes the declared payload files; the manifest itself is packaged "
                    "as the verifier entry point and is intentionally not self-hashed.\n"
                    "Mirrored raw Slurm logs are diagnostic run records and are not part of the "
                    "anonymous submission artifact.\n"
                    "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . "
                    "--write-submission-zip bgr-aaai27-anonymous.zip\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "provenance recipes|remote input paths"):
                check_submission_scope_docs(root)

    def test_submission_scope_docs_rejects_self_hash_ambiguous_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                (
                    "Only files declared by `submission_manifest.json` should be treated as the "
                    "anonymous submission artifact.\n"
                    "Mirrored raw Slurm logs are diagnostic run records and are not part of the "
                    "anonymous submission artifact.\n"
                    "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root . "
                    "--write-submission-zip bgr-aaai27-anonymous.zip\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale anonymous submission scope"):
                check_submission_scope_docs(root)

    def test_root_readme_submission_framing_accepts_current_package_description(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "This repository contains the anonymous AAAI-27 submission package.",
                        "It includes a SHA-256 submission manifest.",
                        "The main evidence includes a 30-seed synthetic mechanism check, active-estimator validation, a completed 30-seed procedural grid-margin full-baseline comparison, a held-out grid replication, a 30-seed robot-suffix coverage comparison, a held-out suffix full-baseline replication, and a held-out suffix BGR-vs-uniform replication.",
                        "The package also includes a 30-seed suffix stress sweep over teacher quality, clutter, feasibility, and boundary sharpness.",
                        "It also includes a 30-seed robot-suffix coverage comparison.",
                        "OpenVLA/LIBERO results are included as recovery-curve, selection, and data-plumbing audits.",
                        "The packaged action-label/TFDS plumbing audit validates 2,048-transition matched BGR/random exports with 7D actions and 8D state.",
                        "They are framed rather than robotics fine-tuning claims.",
                        "## Reviewer Navigation",
                        "Start with `paper/main.pdf` for the anonymous manuscript.",
                        "results/README.md#submission-evidence-index",
                        "The primary evidence is the 30-seed synthetic mechanism check, the active-estimator validation, the 30-seed grid-margin comparison, the held-out grid replication, the 30-seed robot-suffix coverage comparison, the held-out suffix full-baseline replication, the held-out suffix BGR-vs-uniform replication, the suffix stress sweep, and `paper/figures/significance_tests.csv`.",
                        "OpenVLA/LIBERO entries are scoped audits and should not be read as robotics fine-tuning claims.",
                        "For the primary paired comparisons, competing methods share experiment configs, replayable-state pools, evaluation radius grids, learner/update budgets, and paired seeds.",
                        "the intended intervention is the replay state/radius selection rule.",
                        "Baseline rows in the generated tables come from the same scripts and summary artifacts listed in the evidence index.",
                        "The completed 30-seed full-baseline config is the primary grid comparison.",
                        "See results/README.md for the packaged run ledger.",
                        "## Claim-Evidence Map",
                        "| Paper claim | Primary artifact evidence | Verification hook |",
                        "Controlled synthetic recovery-margin training validates the intended BGR sampler before higher-cost runs.",
                        "results/toy_15seed_v1/summary.csv",
                        "results/toy_30seed_v1/summary.csv",
                        "Boundary-centered replay expands recovery margins in the main procedural setting.",
                        "results/grid_margin_full_30seed_v1/summary.csv",
                        "results/grid_margin_full_replication_30seed_v1/summary.csv",
                        "Active boundary probing estimates useful critical radii at a small fixed rollout budget.",
                        "paper/figures/estimator_stats.csv",
                        "paper/figures/estimator_table.tex",
                        "results/estimator_pair_30seed_v1/summary.csv",
                        "Radius-level boundary sampling is the important BGR ablation in the grid-margin benchmark.",
                        "results/grid_margin_ablation_15seed_v1/summary.csv",
                        "results/grid_margin_ablation_30seed_v1/summary.csv",
                        "results/grid_margin_ablation_replication_30seed_v1/summary.csv",
                        "Coverage-aware BGR-Suffix is positive manipulation-style evidence but not a final robotics claim.",
                        "results/suffix_coverage_full_30seed_v1/summary.csv",
                        "results/suffix_coverage_full_replication_30seed_v1/summary.csv",
                        "results/suffix_strategy_coverage_30seed_v1/summary.csv",
                        "results/suffix_strategy_coverage_replication_30seed_v1/summary.csv",
                        "results/suffix_strategy_ablation_30seed_v1/summary.csv",
                        "results/suffix_stress_sensitivity_30seed_v1/summary.csv",
                        "paper/figures/suffix_stress_sensitivity_stats.csv",
                        "The learned-policy OpenVLA/LIBERO path is an audit, not a robotics fine-tuning claim.",
                        "packaged OpenVLA-OFT audit summaries listed below.",
                        "Grid-margin robustness/scope diagnostic artifacts",
                        "paper/figures/grid_margin_learning_curve_stats.csv",
                        "results/grid_margin_full_15seed_v1/results.json",
                        "paper/figures/grid_margin_target_sensitivity_stats.csv",
                        "results/grid_margin_target_sensitivity_15seed_v1/summary.csv",
                        "results/grid_margin_target_sensitivity_30seed_v1/summary.csv",
                        "paper/figures/grid_margin_learning_rate_sensitivity_stats.csv",
                        "results/grid_margin_learning_rate_sensitivity_15seed_v1/summary.csv",
                        "results/grid_margin_learning_rate_sensitivity_30seed_v1/summary.csv",
                        "paper/figures/grid_margin_regime_sensitivity_stats.csv",
                        "results/grid_margin_regime_sensitivity_15seed_v1/summary.csv",
                        "results/grid_margin_regime_sensitivity_30seed_v1/summary.csv",
                        "The packaged `obstacle_prob`/`max_offset` sweep mostly reproduces the nominal margin dynamics.",
                        "rendered table uses the 30-seed regime diagnostic source",
                        "paper/figures/grid_margin_stress_sensitivity_stats.csv",
                        "results/grid_margin_stress_sensitivity_15seed_v1/summary.csv",
                        "results/grid_margin_stress_sensitivity_30seed_v1/summary.csv",
                        "paper/figures/suffix_stress_sensitivity_stats.csv",
                        "results/suffix_stress_sensitivity_15seed_v1/summary.csv",
                        "results/suffix_stress_sensitivity_30seed_v1/summary.csv",
                        "OpenVLA-OFT packaged audit summaries",
                        "OpenVLA recovery audit source",
                        "OpenVLA selection audit source",
                        "OpenVLA action-label/TFDS validation source",
                        "official-checkpoint sanity audit",
                        "1,000-step balanced2048 data-plumbing audit",
                        "p1024 clean adaptation audit",
                        "p1024 original perturbation audit",
                        "p1024 offset-3 perturbation audit",
                        "p2048 clean adaptation audit",
                        "p2048 original perturbation audit",
                        "p2048 offset-3 perturbation audit",
                        "p2048 10-trial perturbation variance audit",
                        "p2048 full-goal clean identity audit",
                        "p2048 full-goal visual perturbation audit",
                        "p2048 300-step image-augmentation continuation audit",
                        "paper/figures/openvla_stats.csv",
                        "results/libero_openvla_recovery_v1/summary.csv",
                        "results/libero_openvla_boundary_selection_balanced_v1/aggregate.csv",
                        "results/openvla_action_tfds_validation_v1/summary.json",
                        "results/openvla_oft_sanity_eval_sanity_v1/summary.csv",
                        "results/openvla_oft_eval_balanced2048_step1000_v1/summary.csv",
                        "results/openvla_oft_clean_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv",
                        "results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_fullgoal10x10_v1/summary.csv",
                        "results/openvla_oft_perturb_eval_cleanmix_p2048_step50300_lr5em7_identitylora_imageaug_officialtrainstats_fullgoal10x10_v1/summary.csv",
                        "results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv",
                        "results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1/summary.csv",
                        "results/openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_10trials_v1/summary.csv",
                        "paper-negative scale-diagnostic outputs out of the anonymous manifest.",
                        "## AAAI Sources",
                        "The official AAAI-27 page lists the 2026 author-submission timetable.",
                        "February 16--23, 2027 conference.",
                        "abstracts are due July 21, 2026.",
                        "full papers July 28, 2026.",
                        "supplementary material/code July 31, 2026.",
                        "paper/AuthorKit27",
                        "https://aaai.org/authorkit27/",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_root_readme_submission_framing(root),
                [f"{readme}: top-level submission framing ok"],
            )

    def test_root_readme_submission_framing_rejects_bounded_openvla_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text("OpenVLA/LIBERO entries are bounded audits\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale top-level README framing"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_rejects_bounded_claim_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text("The top-level claim stays bounded.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale top-level README framing"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_rejects_unscoped_openvla_source_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text("OpenVLA recovery-stat source\nOpenVLA selection-stat source\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale top-level README framing"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_rejects_scaffold_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "This repository scaffolds the AAAI-27 submission.",
                        "paper/ AAAI-27 manuscript skeleton and official AuthorKit27",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale top-level README framing"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_rejects_current_strongest_grid_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "The completed 30-seed full-baseline config is the current strongest grid comparison.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale top-level README framing"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_rejects_current_regime_sweep_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "The current `obstacle_prob`/`max_offset` sweep mostly reproduces the nominal margin dynamics.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale top-level README framing"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_rejects_current_run_ledger_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "See results/README.md for the current run ledger.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale top-level README framing"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_rejects_unscoped_openvla_adaptation_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "- p1024 clean adaptation: `results/openvla_oft_goal_adapt_eval_cleanmix_p1024_step50100_lr1em6_identitylora_officialtrainstats_v1/summary.csv`",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale top-level README framing"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_rejects_stale_aaai_timetable_year(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "The official AAAI-27 page lists the 2027 timetable and links the AAAI-27 author kit.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "2027 timetable"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_requires_reviewer_navigation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "This repository contains the anonymous AAAI-27 submission package.",
                        "It includes a SHA-256 submission manifest.",
                        "The main evidence includes a 30-seed synthetic mechanism check, active-estimator validation, a completed 30-seed procedural grid-margin full-baseline comparison, a held-out grid replication, a 30-seed robot-suffix coverage comparison, a held-out suffix full-baseline replication, and a held-out suffix BGR-vs-uniform replication.",
                        "It also includes a 30-seed robot-suffix coverage comparison.",
                        "OpenVLA/LIBERO results are included as recovery-curve, selection, and data-plumbing audits.",
                        "They are framed rather than robotics fine-tuning claims.",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Reviewer Navigation"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_requires_pairing_invariants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "This repository contains the anonymous AAAI-27 submission package.",
                        "It includes a SHA-256 submission manifest.",
                        "The main evidence includes a 30-seed synthetic mechanism check, active-estimator validation, a completed 30-seed procedural grid-margin full-baseline comparison, a held-out grid replication, a 30-seed robot-suffix coverage comparison, a held-out suffix full-baseline replication, and a held-out suffix BGR-vs-uniform replication.",
                        "It also includes a 30-seed robot-suffix coverage comparison.",
                        "OpenVLA/LIBERO results are included as recovery-curve, selection, and data-plumbing audits.",
                        "They are framed rather than robotics fine-tuning claims.",
                        "## Reviewer Navigation",
                        "Start with `paper/main.pdf` for the anonymous manuscript.",
                        "results/README.md#submission-evidence-index",
                        "The primary evidence is the 30-seed synthetic mechanism check, the active-estimator validation, the 30-seed grid-margin comparison, the held-out grid replication, the 30-seed robot-suffix coverage comparison, the held-out suffix full-baseline replication, the held-out suffix BGR-vs-uniform replication, and `paper/figures/significance_tests.csv`.",
                        "OpenVLA/LIBERO entries are scoped audits and should not be read as robotics fine-tuning claims.",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "paired comparisons|replay state/radius"):
                check_root_readme_submission_framing(root)

    def test_root_readme_submission_framing_requires_claim_evidence_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "This repository contains the anonymous AAAI-27 submission package.",
                        "It includes a SHA-256 submission manifest.",
                        "The main evidence includes a 30-seed synthetic mechanism check, active-estimator validation, a completed 30-seed procedural grid-margin full-baseline comparison, a held-out grid replication, a 30-seed robot-suffix coverage comparison, a held-out suffix full-baseline replication, and a held-out suffix BGR-vs-uniform replication.",
                        "It also includes a 30-seed robot-suffix coverage comparison.",
                        "OpenVLA/LIBERO results are included as recovery-curve, selection, and data-plumbing audits.",
                        "They are framed rather than robotics fine-tuning claims.",
                        "## Reviewer Navigation",
                        "Start with `paper/main.pdf` for the anonymous manuscript.",
                        "results/README.md#submission-evidence-index",
                        "The primary evidence is the 30-seed synthetic mechanism check, the active-estimator validation, the 30-seed grid-margin comparison, the held-out grid replication, the 30-seed robot-suffix coverage comparison, the held-out suffix full-baseline replication, the held-out suffix BGR-vs-uniform replication, and `paper/figures/significance_tests.csv`.",
                        "OpenVLA/LIBERO entries are scoped audits and should not be read as robotics fine-tuning claims.",
                        "For the primary paired comparisons, competing methods share experiment configs, replayable-state pools, evaluation radius grids, learner/update budgets, and paired seeds.",
                        "the intended intervention is the replay state/radius selection rule.",
                        "Baseline rows in the generated tables come from the same scripts and summary artifacts listed in the evidence index.",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Claim-Evidence Map|Boundary-centered replay"):
                check_root_readme_submission_framing(root)

    def test_root_readme_verification_commands_accepts_full_gate_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "## Verification Commands",
                        "python3 -m pip install -e .",
                        "PYTHONPATH=src:. python3 -m unittest discover -s tests",
                        "PYTHONPATH=src:. python3 scripts/check_paper_claims.py --paper paper/main.tex --results-dir results --figures-dir paper/figures",
                        "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .",
                        "The package gate verifies rendered PDFs, PDF metadata hygiene, required artifacts, generated table synchronization, double-blind hygiene, README framing, and the SHA-256 manifest.",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_root_readme_verification_commands(root),
                [f"{readme}: top-level verification commands ok"],
            )

    def test_root_readme_verification_commands_requires_claim_checker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "## Verification Commands",
                        "python3 -m pip install -e .",
                        "PYTHONPATH=src:. python3 -m unittest discover -s tests",
                        "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .",
                        "The package gate verifies rendered PDFs, PDF metadata hygiene, required artifacts, generated table synchronization, double-blind hygiene, README framing, and the SHA-256 manifest.",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "check_paper_claims"):
                check_root_readme_verification_commands(root)

    def test_main_writes_required_manifest_without_running_package_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in required_submission_files():
                if relative == SUBMISSION_MANIFEST:
                    continue
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{relative}\n", encoding="utf-8")

            buffer = io.StringIO()
            with mock.patch(
                "sys.argv",
                [
                    "check_submission_package.py",
                    "--root",
                    str(root),
                    "--write-required-manifest",
                ],
            ), mock.patch(
                "scripts.check_submission_package.check_package",
                side_effect=AssertionError("check_package should not run"),
            ), redirect_stdout(buffer):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(buffer.getvalue(), f"wrote {SUBMISSION_MANIFEST}\n")
            self.assertEqual(
                check_submission_manifest(root),
                [f"{SUBMISSION_MANIFEST}: required artifact SHA-256 manifest ok"],
            )

    def write_required_tree_with_manifest(self, root: Path) -> None:
        for relative in required_submission_files():
            if relative == SUBMISSION_MANIFEST:
                continue
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{relative}\n", encoding="utf-8")
        (root / SUBMISSION_MANIFEST).write_text(canonical_json(submission_manifest_payload(root)), encoding="utf-8")

    def test_write_submission_zip_contains_only_required_artifacts_deterministically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_required_tree_with_manifest(root)
            archive_path = root / "dist" / "bgr-aaai27-anonymous.zip"

            count = write_submission_zip(root, archive_path)

            required = required_submission_files()
            self.assertEqual(count, len(required))
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.namelist(), required)
                self.assertEqual(archive.read("README.md"), b"README.md\n")
                for info in archive.infolist():
                    self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))

    def test_write_submission_zip_rejects_required_artifact_output_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_required_tree_with_manifest(root)

            with self.assertRaisesRegex(ValueError, "required artifact path: paper/main.pdf"):
                write_submission_zip(root, root / "paper" / ".." / "paper" / "main.pdf")

    def test_main_writes_submission_zip_after_package_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_required_tree_with_manifest(root)
            archive_path = root / "bgr-aaai27-anonymous.zip"
            buffer = io.StringIO()

            with mock.patch(
                "sys.argv",
                [
                    "check_submission_package.py",
                    "--root",
                    str(root),
                    "--write-submission-zip",
                    str(archive_path),
                ],
            ), mock.patch(
                "scripts.check_submission_package.check_package",
                return_value=["package ok"],
            ), redirect_stdout(buffer):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn(f"wrote {archive_path}", buffer.getvalue())
            self.assertTrue(archive_path.exists())

    def test_required_text_artifacts_double_blind_allows_vendored_author_kit_thanks_macro(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in required_text_artifact_files():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("anonymous artifact text\n", encoding="utf-8")
            thanks_macro = next(pattern for pattern in DOUBLE_BLIND_FORBIDDEN_PATTERNS if pattern.startswith("\\"))
            (root / "paper" / "AuthorKit27" / "aaai2027.sty").write_text(
                f"\\newcommand{{{thanks_macro}}}[1]{{}}\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_required_text_artifacts_double_blind(root),
                ["required text artifacts are double-blind safe ok"],
            )

    def test_openvla_bridge_artifacts_reject_identity_path_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in OPENVLA_BRIDGE_ARTIFACTS + OPENVLA_BRIDGE_TEST_ARTIFACTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("#!/usr/bin/env bash\nREMOTE_PROJECT=/work/anonymous/bgr\n", encoding="utf-8")
            leaked = root / OPENVLA_BRIDGE_TEST_ARTIFACTS[0]
            leaked.write_text(f"#!/usr/bin/env bash\nREMOTE_PROJECT={self.author_pattern('/work/')}bgr\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "double-blind leak"):
                check_openvla_bridge_scripts(root)

    def test_openvla_bridge_artifacts_accept_anonymous_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in OPENVLA_BRIDGE_ARTIFACTS + OPENVLA_BRIDGE_TEST_ARTIFACTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("#!/usr/bin/env bash\nREMOTE_PROJECT=/work/anonymous/bgr\n", encoding="utf-8")

            self.assertEqual(
                check_openvla_bridge_scripts(root),
                ["OpenVLA bridge scripts/tests are double-blind safe ok"],
            )

    def test_required_git_tracked_accepts_submission_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in required_submission_files():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            with mock.patch(
                "scripts.check_submission_package.git_tracked_files",
                return_value=set(required_submission_files()),
            ):
                self.assertEqual(
                    check_required_git_tracked(root),
                    ["required submission artifacts are tracked by git ok"],
                )

    def test_required_git_tracked_rejects_untracked_submission_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            required = required_submission_files()
            missing_from_git = "configs/grid_margin_full_30seed.yaml"
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            with mock.patch(
                "scripts.check_submission_package.git_tracked_files",
                return_value=set(required) - {missing_from_git},
            ):
                with self.assertRaisesRegex(ValueError, "not tracked by git.*configs/grid_margin_full_30seed.yaml"):
                    check_required_git_tracked(root)

    def test_missing_required_files_reports_absent_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            required = required_submission_files()
            missing = "configs/grid_margin_full_30seed.yaml"
            for relative in required:
                if relative == missing:
                    continue
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            self.assertEqual(missing_required_files(root), [missing])

    def test_untracked_required_files_reports_git_subset(self):
        required = required_submission_files()
        untracked = "configs/grid_margin_full_30seed.yaml"

        with mock.patch(
            "scripts.check_submission_package.git_tracked_files",
            return_value=set(required) - {untracked},
        ):
            self.assertEqual(untracked_required_files(Path(".")), [untracked])

    def test_main_lists_required_artifacts_without_running_package_checks(self):
        buffer = io.StringIO()

        with mock.patch("sys.argv", ["check_submission_package.py", "--list-required-artifacts"]), mock.patch(
            "scripts.check_submission_package.check_package",
            side_effect=AssertionError("check_package should not run"),
        ), redirect_stdout(buffer):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertIn("paper/main.pdf\n", buffer.getvalue())

    def test_main_lists_untracked_required_artifacts_without_mutating_git(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            required = required_submission_files()
            untracked = "configs/grid_margin_full_30seed.yaml"
            for relative in required:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")

            buffer = io.StringIO()
            with mock.patch(
                "sys.argv",
                [
                    "check_submission_package.py",
                    "--root",
                    str(root),
                    "--list-untracked-required-artifacts",
                ],
            ), mock.patch(
                "scripts.check_submission_package.git_tracked_files",
                return_value=set(required) - {untracked},
            ), mock.patch(
                "scripts.check_submission_package.check_package",
                side_effect=AssertionError("check_package should not run"),
            ), redirect_stdout(buffer):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(buffer.getvalue(), f"{untracked}\n")

    def test_main_list_untracked_required_artifacts_rejects_missing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            buffer = io.StringIO()

            with mock.patch(
                "sys.argv",
                [
                    "check_submission_package.py",
                    "--root",
                    str(root),
                    "--list-untracked-required-artifacts",
                ],
            ), redirect_stdout(buffer):
                exit_code = main()

            self.assertEqual(exit_code, 1)
            self.assertIn("required submission artifact(s) are missing", buffer.getvalue())

    def test_technical_page_limit_rejects_late_embedded_checklist(self):
        with mock.patch(
            "scripts.check_submission_package.pdf_metadata",
            return_value=PdfMetadata(pages=9, page_size="612 x 792 pts (letter)", file_size=1000),
        ), mock.patch(
            "scripts.check_submission_package.pdf_page_text",
            side_effect=["", "", "", "", "", "", "", "", "Reproducibility Checklist"],
        ):
            with self.assertRaisesRegex(ValueError, "technical content uses 8 pages"):
                check_technical_page_limit(Path("paper/main.pdf"))

    def test_technical_page_limit_accepts_checklist_after_five_pages(self):
        with mock.patch(
            "scripts.check_submission_package.pdf_metadata",
            return_value=PdfMetadata(pages=7, page_size="612 x 792 pts (letter)", file_size=1000),
        ), mock.patch(
            "scripts.check_submission_package.pdf_page_text",
            side_effect=["", "", "", "", "", "Reproducibility Checklist", ""],
        ):
            self.assertEqual(
                check_technical_page_limit(Path("paper/main.pdf")),
                ["paper/main.pdf: technical content uses 5/7 pages before checklist"],
            )

    def test_results_ledger_rejects_stale_completed_p2048_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text("## In-Progress OpenVLA-OFT p2048 Clean-Mix Scale-Up\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale p2048 ledger"):
                check_results_ledger(root)

    def test_results_ledger_accepts_completed_p2048_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p2048 Clean-Mix Scale-Up",
                        "TFDS roots:",
                        "| Matched random | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                        "## Completed OpenVLA-OFT p2048 Offset-3 Perturbation Eval",
                        "Corrected submitted command",
                        "| Random-balanced | 1.0000 | 0.9714 | 0.9714 | 0.5714 | 0.9714 | 0.8714 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                        "## Completed OpenVLA-OFT p1024 Offset-3 Perturbation Eval",
                        "Submitted command",
                        "| Random-balanced | 1.0000 | 1.0000 | 0.9429 | 0.5429 | 0.9429 | 0.8571 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                        "## Completed OpenVLA-OFT p1024 Perturbation-Mix Diagnostic",
                        "Submitted prep command",
                        "Submitted TFDS continuation command",
                        "Submitted clean eval command",
                        "Submitted visual-perturbation eval command",
                        "Submitted official visual-perturbation comparator",
                        "| BGR-boundary | 0.9333 | 0.9333 | 0.9333 | 0.5333 | 0.9333 | 0.8333 |",
                        "| Random-balanced | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(check_results_ledger(root), [f"{ledger}: p2048 completion ledger ok"])

    def test_results_ledger_rejects_stale_completed_p2048_queue_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p2048 Clean-Mix Scale-Up",
                        "TFDS roots:",
                        "| Matched random | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                        "Queued prep/TFDS command",
                        "Queued clean adaptation/eval continuation",
                        "Queued perturbation eval command",
                        "Expected TFDS roots",
                        "## Completed OpenVLA-OFT p2048 Offset-3 Perturbation Eval",
                        "Corrected submitted command",
                        "| Random-balanced | 1.0000 | 0.9714 | 0.9714 | 0.5714 | 0.9714 | 0.8714 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale completed p2048 ledger"):
                check_results_ledger(root)

    def test_results_ledger_rejects_stale_completed_p2048_self_instruction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p2048 Clean-Mix Scale-Up",
                        "TFDS roots:",
                        "| Matched random | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                        "OpenVLA-OFT should remain a diagnostic audit in the paper.",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale p2048 ledger|stale completed p2048 ledger"):
                check_results_ledger(root)

    def test_results_ledger_rejects_stale_completed_p2048_offset3_queue_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p2048 Clean-Mix Scale-Up",
                        "TFDS roots:",
                        "| Matched random | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                        "## Completed OpenVLA-OFT p2048 Offset-3 Perturbation Eval",
                        "Corrected submitted command",
                        "| Random-balanced | 1.0000 | 0.9714 | 0.9714 | 0.5714 | 0.9714 | 0.8714 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                        "Queued and completed on 2026-06-03",
                        "Corrected queued command",
                        "## Completed OpenVLA-OFT p1024 Offset-3 Perturbation Eval",
                        "Submitted command",
                        "| Random-balanced | 1.0000 | 1.0000 | 0.9429 | 0.5429 | 0.9429 | 0.8571 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                        "## Completed OpenVLA-OFT p1024 Perturbation-Mix Diagnostic",
                        "Submitted prep command",
                        "Submitted TFDS continuation command",
                        "Submitted clean eval command",
                        "Submitted visual-perturbation eval command",
                        "Submitted official visual-perturbation comparator",
                        "| BGR-boundary | 0.9333 | 0.9333 | 0.9333 | 0.5333 | 0.9333 | 0.8333 |",
                        "| Random-balanced | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale completed p2048 offset-3 ledger"):
                check_results_ledger(root)

    def test_results_ledger_rejects_stale_completed_p1024_offset_queue_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p2048 Clean-Mix Scale-Up",
                        "TFDS roots:",
                        "| Matched random | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                        "## Completed OpenVLA-OFT p2048 Offset-3 Perturbation Eval",
                        "Corrected submitted command",
                        "| Random-balanced | 1.0000 | 0.9714 | 0.9714 | 0.5714 | 0.9714 | 0.8714 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                        "## Completed OpenVLA-OFT p1024 Offset-3 Perturbation Eval",
                        "Submitted command",
                        "| Random-balanced | 1.0000 | 1.0000 | 0.9429 | 0.5429 | 0.9429 | 0.8571 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                        "Queued on 2026-06-02 to test whether the current p1024 one-episode occlusion",
                        "Queued command",
                        "## Completed OpenVLA-OFT p1024 Perturbation-Mix Diagnostic",
                        "Submitted prep command",
                        "Submitted TFDS continuation command",
                        "Submitted clean eval command",
                        "Submitted visual-perturbation eval command",
                        "Submitted official visual-perturbation comparator",
                        "| BGR-boundary | 0.9333 | 0.9333 | 0.9333 | 0.5333 | 0.9333 | 0.8333 |",
                        "| Random-balanced | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale completed p1024 offset-3 ledger"):
                check_results_ledger(root)

    def test_results_ledger_rejects_stale_completed_p1024_mix_queue_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p2048 Clean-Mix Scale-Up",
                        "TFDS roots:",
                        "| Matched random | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                        "## Completed OpenVLA-OFT p2048 Offset-3 Perturbation Eval",
                        "Corrected submitted command",
                        "| Random-balanced | 1.0000 | 0.9714 | 0.9714 | 0.5714 | 0.9714 | 0.8714 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                        "## Completed OpenVLA-OFT p1024 Offset-3 Perturbation Eval",
                        "Submitted command",
                        "| Random-balanced | 1.0000 | 1.0000 | 0.9429 | 0.5429 | 0.9429 | 0.8571 |",
                        "| Official OpenVLA-OFT | 174 / 200 | 0.8700 |",
                        "## Completed OpenVLA-OFT p1024 Perturbation-Mix Diagnostic",
                        "Submitted prep command",
                        "Submitted TFDS continuation command",
                        "Submitted clean eval command",
                        "Submitted visual-perturbation eval command",
                        "Submitted official visual-perturbation comparator",
                        "| BGR-boundary | 0.9333 | 0.9333 | 0.9333 | 0.5333 | 0.9333 | 0.8333 |",
                        "| Random-balanced | 0.9333 | 0.9333 | 0.9333 | 0.4000 | 0.9333 | 0.8000 |",
                        "Queued on 2026-06-02 after the completed p512 diagnostic",
                        "Queued prep command",
                        "Queued TFDS continuation command",
                        "Queued clean eval command",
                        "Queued visual-perturbation eval command",
                        "Queued official visual-perturbation comparator",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale completed p1024 perturbation-mix ledger"):
                check_results_ledger(root)

    def test_paper_facing_openvla_staleness_rejects_latest_p1024(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "paper" / "main.tex"
            paper.parent.mkdir(parents=True, exist_ok=True)
            paper.write_text("In the latest p1024 clean-mix diagnostic, BGR remains diagnostic.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale paper-facing OpenVLA scale wording"):
                check_paper_facing_openvla_staleness(root)

    def test_paper_facing_openvla_staleness_rejects_current_p1024_superlative(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_readme = root / "paper" / "README.md"
            paper_readme.parent.mkdir(parents=True, exist_ok=True)
            paper_readme.write_text(
                "The p1024 run is the strongest current OpenVLA-OFT diagnostic.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale paper-facing OpenVLA scale wording"):
                check_paper_facing_openvla_staleness(root)

    def test_paper_facing_openvla_staleness_rejects_old_clean_sentence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "paper" / "main.tex"
            paper.parent.mkdir(parents=True, exist_ok=True)
            paper.write_text(
                "BGR-selected and matched-random adaptation score 14/15 clean episodes for both BGR and matched random.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale paper-facing OpenVLA scale wording"):
                check_paper_facing_openvla_staleness(root)

    def test_paper_facing_openvla_staleness_rejects_old_failed_pilot_framing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "paper" / "main.tex"
            paper.parent.mkdir(parents=True, exist_ok=True)
            paper.write_text(
                "A 1,000-step OpenVLA-OFT pilot completes the chain but scores 0/15.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale paper-facing OpenVLA scale wording"):
                check_paper_facing_openvla_staleness(root)

    def test_paper_facing_openvla_staleness_accepts_neutral_scale_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper = root / "paper" / "main.tex"
            readme = root / "README.md"
            paper_readme = root / "paper" / "README.md"
            paper.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text("At p2048, BGR and matched random again tie clean.", encoding="utf-8")
            paper.write_text("In the p1024 clean-mix diagnostic, BGR remains diagnostic.", encoding="utf-8")
            paper_readme.write_text("The pooled p1024 visual-perturbation evidence is diagnostic.", encoding="utf-8")

            self.assertEqual(check_paper_facing_openvla_staleness(root), ["paper-facing OpenVLA scale wording ok"])

    def test_p4096_status_rejects_missing_in_progress_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "results" / "README.md"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text("No p4096 section yet.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing in-progress p4096 ledger"):
                check_p4096_status(root)

    def test_p4096_status_rejects_paper_facing_claim_before_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "results" / "README.md"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(
                "\n".join(
                    [
                        "## In-Progress OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Treat it as a falsification/scale diagnostic",
                        "763725  bgr-cleanmix-prep-p4096",
                        "Expected local collection paths after completion",
                    ]
                ),
                encoding="utf-8",
            )
            paper = root / "paper" / "main.tex"
            paper.parent.mkdir(parents=True, exist_ok=True)
            paper.write_text("A p4096 result improves OpenVLA.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "p4096 is incomplete"):
                check_p4096_status(root)

    def test_p4096_status_accepts_in_progress_diagnostic_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "results" / "README.md"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(
                "\n".join(
                    [
                        "## In-Progress OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Treat it as a falsification/scale diagnostic",
                        "763725  bgr-cleanmix-prep-p4096",
                        "Expected local collection paths after completion",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(check_p4096_status(root), [f"{ledger}: p4096 in-progress diagnostic ledger ok"])

    def test_p4096_status_keeps_partial_summaries_in_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            goal.parent.mkdir(parents=True, exist_ok=True)
            perturb.parent.mkdir(parents=True, exist_ok=True)
            goal.write_text("method,success_rate\nbgr,0.9333\n", encoding="utf-8")
            perturb.write_text("method,perturbation,success_rate\nofficial,identity,0.9333\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## In-Progress OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Treat it as a falsification/scale diagnostic",
                        "763725  bgr-cleanmix-prep-p4096",
                        "Expected local collection paths after completion",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(check_p4096_status(root), [f"{ledger}: p4096 in-progress diagnostic ledger ok"])

    def test_p4096_status_requires_completed_ledger_for_complete_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            goal.parent.mkdir(parents=True, exist_ok=True)
            perturb.parent.mkdir(parents=True, exist_ok=True)
            goal.write_text("method,success_rate\nbgr,0.9333\nrandom,0.9333\n", encoding="utf-8")
            perturb.write_text(
                "method,perturbation,success_rate\n"
                "official,identity,0.9333\n"
                "bgr,identity,0.9333\n"
                "random,identity,0.9333\n",
                encoding="utf-8",
            )
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic\n"
                "Local collection paths:\n",
                encoding="utf-8",
            )

            self.assertEqual(check_p4096_status(root), [f"{ledger}: p4096 completion ledger ok"])

    def test_p4096_status_accepts_completed_diagnostic_with_commonavail_in_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            common_goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            common_perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            for path in [goal, perturb, common_goal, common_perturb]:
                path.parent.mkdir(parents=True, exist_ok=True)
            goal.write_text("method,success_rate\nbgr,0.9333\nrandom,0.9333\n", encoding="utf-8")
            perturb.write_text(
                "method,perturbation,success_rate\n"
                "official,identity,0.9333\n"
                "bgr,identity,0.9333\n"
                "random,identity,0.9333\n",
                encoding="utf-8",
            )
            common_goal.write_text("method,success_rate\nbgr,0.9333\n", encoding="utf-8")
            common_perturb.write_text("method,perturbation,success_rate\nofficial,identity,0.9333\n", encoding="utf-8")
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Local collection paths:",
                        "## In-Progress OpenVLA-OFT p4096 Common-Availability Repair",
                        "Expected common-availability collection paths after completion",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_p4096_status(root),
                [f"{ledger}: p4096 diagnostic complete; common-availability repair in progress ok"],
            )

    def test_p4096_status_blocks_paper_claims_until_commonavail_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            goal.parent.mkdir(parents=True, exist_ok=True)
            perturb.parent.mkdir(parents=True, exist_ok=True)
            goal.write_text("method,success_rate\nbgr,0.9333\nrandom,0.9333\n", encoding="utf-8")
            perturb.write_text(
                "method,perturbation,success_rate\n"
                "official,identity,0.9333\n"
                "bgr,identity,0.9333\n"
                "random,identity,0.9333\n",
                encoding="utf-8",
            )
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Local collection paths:",
                        "## In-Progress OpenVLA-OFT p4096 Common-Availability Repair",
                        "Expected common-availability collection paths after completion",
                    ]
                ),
                encoding="utf-8",
            )
            paper = root / "paper" / "main.tex"
            paper.parent.mkdir(parents=True, exist_ok=True)
            paper.write_text("The p4096 common-availability comparison improves OpenVLA.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "p4096 common-availability repair is incomplete"):
                check_p4096_status(root)

    def test_p4096_status_accepts_paper_negative_completed_commonavail_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            common_goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            common_perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            for path in [goal, perturb, common_goal, common_perturb]:
                path.parent.mkdir(parents=True, exist_ok=True)
            goal.write_text("method,success_rate\nbgr,0.9333\nrandom,0.9333\n", encoding="utf-8")
            perturb.write_text(
                "method,perturbation,success_rate\n"
                "official,identity,0.9333\n"
                "bgr,identity,0.9333\n"
                "random,identity,0.9333\n",
                encoding="utf-8",
            )
            common_goal.write_text("method,success_rate\nbgr,0.9333\nrandom,1.0000\n", encoding="utf-8")
            common_perturb.write_text(
                "method,perturbation,episodes,successes,success_rate\n"
                "official,identity,15,14,0.9333\n"
                "official,blur,15,14,0.9333\n"
                "bgr,identity,15,14,0.9333\n"
                "bgr,blur,15,14,0.9333\n"
                "random,identity,15,15,1.0000\n"
                "random,blur,15,15,1.0000\n",
                encoding="utf-8",
            )
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Local collection paths:",
                        "## Completed OpenVLA-OFT p4096 Common-Availability Repair",
                        "BGR clean is 14/15 = 0.9333; random clean is 15/15 = 1.0000",
                        "Mean perturbed success is 0.8167 for official, 0.8167 for BGR, and 0.8333 for random",
                        "Common-availability collection paths:",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_p4096_status(root),
                [f"{ledger}: p4096 diagnostic and paper-negative common-availability completion ledger ok"],
            )

    def test_p4096_status_accepts_paper_negative_ledger_without_packaged_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "results" / "README.md"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Local collection paths:",
                        "## Completed OpenVLA-OFT p4096 Common-Availability Repair",
                        "BGR clean is 14/15 = 0.9333; random clean is 15/15 = 1.0000.",
                        "Mean perturbed success is 0.8167 for official, 0.8167 for BGR, and 0.8333 for random.",
                        "Common-availability collection paths:",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_p4096_status(root),
                [f"{ledger}: p4096 paper-negative completion ledger retained without packaged summaries ok"],
            )

    def test_p4096_status_rejects_stale_completed_commonavail_queue_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "results" / "README.md"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Local collection paths:",
                        "## Completed OpenVLA-OFT p4096 Common-Availability Repair",
                        "BGR clean is 14/15 = 0.9333; random clean is 15/15 = 1.0000.",
                        "Mean perturbed success is 0.8167 for official, 0.8167 for BGR, and 0.8333 for random.",
                        "Common-availability collection paths:",
                        "Queued common-availability chain:",
                        "Expected common-availability collection paths after completion:",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale completed p4096 common-availability ledger"):
                check_p4096_status(root)

    def test_p4096_status_rejects_stale_completed_scale_queue_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            goal.parent.mkdir(parents=True, exist_ok=True)
            perturb.parent.mkdir(parents=True, exist_ok=True)
            goal.write_text("method,success_rate\nbgr,0.9333\nrandom,0.9333\n", encoding="utf-8")
            perturb.write_text(
                "method,perturbation,success_rate\n"
                "official,identity,0.9333\n"
                "bgr,identity,0.9333\n"
                "random,identity,0.9333\n",
                encoding="utf-8",
            )
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Local collection paths:",
                        "Queued prep/TFDS command",
                        "Queued clean adaptation/eval continuation",
                        "Queued perturbation eval command",
                        "Expected local collection paths after completion",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale completed p4096 ledger"):
                check_p4096_status(root)

    def test_p4096_status_blocks_paper_negative_completed_commonavail_claims(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            common_goal = (
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            common_perturb = (
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p4096_commonavail_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv"
            )
            for path in [goal, perturb, common_goal, common_perturb]:
                path.parent.mkdir(parents=True, exist_ok=True)
            goal.write_text("method,success_rate\nbgr,0.9333\nrandom,0.9333\n", encoding="utf-8")
            perturb.write_text(
                "method,perturbation,success_rate\n"
                "official,identity,0.9333\n"
                "bgr,identity,0.9333\n"
                "random,identity,0.9333\n",
                encoding="utf-8",
            )
            common_goal.write_text("method,success_rate\nbgr,0.9333\nrandom,1.0000\n", encoding="utf-8")
            common_perturb.write_text(
                "method,perturbation,episodes,successes,success_rate\n"
                "official,identity,15,14,0.9333\n"
                "official,blur,15,14,0.9333\n"
                "bgr,identity,15,14,0.9333\n"
                "bgr,blur,15,14,0.9333\n"
                "random,identity,15,15,1.0000\n"
                "random,blur,15,15,1.0000\n",
                encoding="utf-8",
            )
            ledger = root / "results" / "README.md"
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Local collection paths:",
                        "## Completed OpenVLA-OFT p4096 Common-Availability Repair",
                        "BGR clean is 14/15 = 0.9333; random clean is 15/15 = 1.0000",
                        "Mean perturbed success is 0.8167 for official, 0.8167 for BGR, and 0.8333 for random",
                        "Common-availability collection paths:",
                    ]
                ),
                encoding="utf-8",
            )
            paper = root / "paper" / "main.tex"
            paper.parent.mkdir(parents=True, exist_ok=True)
            paper.write_text("The p4096 comparison supports BGR.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "p4096 common-availability result is paper-negative"):
                check_p4096_status(root)

    def test_p4096_status_blocks_paper_negative_ledger_without_packaged_summaries_claims(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "results" / "README.md"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(
                "\n".join(
                    [
                        "## Completed OpenVLA-OFT p4096 Clean-Mix Scale Diagnostic",
                        "Local collection paths:",
                        "## Completed OpenVLA-OFT p4096 Common-Availability Repair",
                        "BGR clean is 14/15 = 0.9333; random clean is 15/15 = 1.0000.",
                        "Mean perturbed success is 0.8167 for official, 0.8167 for BGR, and 0.8333 for random.",
                        "Common-availability collection paths:",
                    ]
                ),
                encoding="utf-8",
            )
            paper = root / "paper" / "main.tex"
            paper.parent.mkdir(parents=True, exist_ok=True)
            paper.write_text("The p4096 comparison supports BGR.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "p4096 common-availability result is paper-negative"):
                check_p4096_status(root)

    def test_root_readme_rejects_stale_openvla_status_after_p2048_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text("A p1024 follow-up is queued under cleanmix_p1024.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale OpenVLA README"):
                check_root_readme_openvla_status(root)

    def test_root_readme_rejects_long_openvla_pilot_section_after_p2048_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "## OpenVLA-OFT Pilot",
                        "The first clean-mix run also collapsed.",
                        "At p2048, BGR and matched random again tie clean at 14/15 each",
                        "original p2048 visual perturbations give BGR 0.8167 vs. 0.8000 for random",
                        "offset-3 follow-up gives BGR 0.8714 vs. 0.8714 for random, with official at 0.8929",
                        "Pooling p2048 gives BGR 0.8550 vs. 0.8500 for random, trailing official at 0.8700",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale OpenVLA README"):
                check_root_readme_openvla_status(root)

    def test_root_readme_accepts_current_openvla_status_after_p2048_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text(
                "\n".join(
                    [
                        "## OpenVLA/LIBERO Audit Summary",
                        "Detailed OpenVLA commands, Slurm IDs, copied artifacts, and summaries are in `results/README.md`.",
                        "The top-level claim is scoped rather than robotics fine-tuning performance.",
                        "The packaged action-label/TFDS plumbing audit validates 2,048-transition matched BGR/random exports with 7D actions and 8D state.",
                        "The packaged useful audit scale is p1024/p2048 clean-mix adaptation.",
                        "At p1024, BGR and matched random tie clean at 14/15",
                        "pooling the original and offset-3 visual perturbation evals gives BGR 0.8550 vs. 0.8400 for random",
                        "At p2048, the full-goal identity audit gives 99/100 clean successes",
                        "The 10-task visual perturbation audit gives BGR 367/400 perturbed successes",
                        "trailing matched random by one episode (368/400)",
                        "The 300-step image-augmentation continuation gives BGR and matched random 368/400 perturbed successes each",
                        "only one episode above official (367/400)",
                        "The 1,000-step low-learning-rate continuation is also negative",
                        "BGR gives 366/400 non-identity perturbation successes",
                        "The follow-up weighted perturbation curriculum is also negative",
                        "matched random reaches 370/400",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_root_readme_openvla_status(root),
                [f"{readme}: OpenVLA p2048 README status ok"],
            )

    def test_root_readme_rejects_current_openvla_audit_scale_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            readme = root / "README.md"
            readme.write_text("The current useful audit scale is p1024/p2048 clean-mix adaptation.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale OpenVLA README"):
                check_root_readme_openvla_status(root)

    def test_paper_readme_rejects_stale_openvla_status_after_p2048_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "final control is 14/15 clean for BGR vs. 15/15 for random",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale OpenVLA paper README"):
                check_paper_readme_openvla_status(root)

    def test_paper_readme_rejects_bounded_openvla_claim_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text("The bridge preserves competence while keeping the claim bounded.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale OpenVLA paper README"):
                check_paper_readme_openvla_status(root)

    def test_paper_readme_accepts_current_openvla_status_after_p2048_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in [
                root
                / "results"
                / "openvla_oft_goal_adapt_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_v1"
                / "summary.csv",
                root
                / "results"
                / "openvla_oft_perturb_eval_cleanmix_p2048_step50100_lr1em6_identitylora_officialtrainstats_offset3_7trials_v1"
                / "summary.csv",
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("method,success_rate\nbgr,1.0\n", encoding="utf-8")
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "\n".join(
                    [
                        "OpenVLA-OFT bridge includes corrected clean-mix diagnostics that preserve competence while keeping the claim scoped",
                        "The 30-seed suffix stress confirmation",
                        "BGR-Coverage beats uniform with 30/0 paired wins on clean success, final object RAUC, transfer RAUC, and AULC in all four stress cases",
                        "The packaged action-label/TFDS plumbing audit validates 2,048-transition matched BGR/random exports with 7D actions and 8D state.",
                        "pooled p1024 visual-perturbation evidence gives BGR 0.8550 vs. 0.8400 for random",
                        "trailing official at 0.8700",
                        "At p2048, the full-goal identity audit gives 99/100 clean successes",
                        "The 10-task visual perturbation audit gives BGR 367/400 perturbed successes",
                        "trailing matched random by one episode (368/400)",
                        "The 300-step image-augmentation continuation gives BGR and matched random 368/400 perturbed successes each",
                        "only one episode above official (367/400)",
                        "The 1,000-step low-learning-rate continuation is also negative",
                        "The follow-up weighted perturbation curriculum is also negative",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_paper_readme_openvla_status(root),
                [f"{readme}: OpenVLA p2048 paper README status ok"],
            )

    def test_paper_readme_verification_commands_accepts_claim_and_package_gates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "\n".join(
                    [
                        "From the repository root, verify the paper-facing claims and complete submission package before submission.",
                        "PYTHONPATH=src:. python3 scripts/check_paper_claims.py --paper paper/main.tex --results-dir results --figures-dir paper/figures",
                        "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_paper_readme_verification_commands(root),
                [f"{readme}: paper README verification commands ok"],
            )

    def test_paper_readme_verification_commands_requires_claim_checker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "\n".join(
                    [
                        "From the repository root, verify the paper-facing claims and complete submission package before submission.",
                        "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "check_paper_claims"):
                check_paper_readme_verification_commands(root)

    def test_paper_readme_artifact_references_accepts_packaged_command_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "\n".join(
                    [
                        "PYTHONPATH=src:. python3 scripts/check_paper_claims.py --paper paper/main.tex --results-dir results --figures-dir paper/figures",
                        "PYTHONPATH=src:. python3 scripts/check_submission_package.py --root .",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_paper_readme_artifact_references(root),
                [f"{readme}: paper README local artifact references are packaged ok"],
            )

    def test_paper_readme_artifact_references_rejects_unpackaged_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "Do not cite results/not_packaged_v1/summary.csv as paper evidence.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "paper README local artifact reference"):
                check_paper_readme_artifact_references(root)

    def test_paper_readme_build_provenance_accepts_included_pdf_gate_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "\n".join(
                    [
                        "The AAAI style requires pdfTeX.",
                        "Tectonic/XeTeX is not a suitable fallback for this AuthorKit.",
                        "Use the package gate below to verify the included PDFs.",
                        "Clean rebuilt PDFs with `qpdf --remove-info --remove-metadata`.",
                        "The gate checks rendered/source claim sync.",
                        "Then rebuild on a pdfTeX-enabled host before replacing the PDFs.",
                        "AAAI rule sources were rechecked on 2026-06-04.",
                        "On that date, the official AAAI-27 page linked the AAAI-27 Author Kit.",
                        "The AAAI-27 page and bundled AAAI-27 AuthorKit are therefore the package authorities for timetable, style, and source-build checks.",
                        "July 21, 2026 abstract.",
                        "July 28, 2026 full-paper.",
                        "July 31, 2026 supplementary material/code deadlines.",
                        "submission-mode anonymity, letter-paper geometry, embedded-font hygiene, embedded checklist placement, and the 7-page technical-content limit as submission-critical checks.",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_paper_readme_build_provenance(root),
                [f"{readme}: paper README build provenance ok"],
            )

    def test_paper_readme_build_provenance_requires_no_tectonic_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "\n".join(
                    [
                        "The AAAI style requires pdfTeX.",
                        "Use the package gate below to verify the included PDFs.",
                        "Clean rebuilt PDFs with `qpdf --remove-info --remove-metadata`.",
                        "The gate checks rendered/source claim sync.",
                        "Then rebuild on a pdfTeX-enabled host before replacing the PDFs.",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Tectonic/XeTeX"):
                check_paper_readme_build_provenance(root)

    def test_paper_readme_build_provenance_requires_current_aaai_rule_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "\n".join(
                    [
                        "The AAAI style requires pdfTeX.",
                        "Tectonic/XeTeX is not a suitable fallback for this AuthorKit.",
                        "Use the package gate below to verify the included PDFs.",
                        "Clean rebuilt PDFs with `qpdf --remove-info --remove-metadata`.",
                        "The gate checks rendered/source claim sync.",
                        "Then rebuild on a pdfTeX-enabled host before replacing the PDFs.",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "AAAI rule sources|July 21, 2026"):
                check_paper_readme_build_provenance(root)

    def test_paper_readme_build_provenance_rejects_current_pdf_verified_claim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "The current PDF was also verified with the standard sequence pdflatex bibtex.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "current PDF was also verified"):
                check_paper_readme_build_provenance(root)

    def test_paper_readme_build_provenance_rejects_latest_rule_source_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "The latest available AAAI main-track submission instructions were still the AAAI-26 instructions.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale paper README PDF provenance"):
                check_paper_readme_build_provenance(root)

    def test_paper_readme_build_provenance_rejects_aaai26_rule_source_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "The AAAI-26 main-track submission instructions were the available main-track rule source during that recheck.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale paper README PDF provenance"):
                check_paper_readme_build_provenance(root)

    def test_paper_readme_submission_framing_accepts_packaged_manuscript_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "\n".join(
                    [
                        "The packaged `main.tex` is an anonymous AAAI submission manuscript.",
                        "It has synthetic mechanism, estimator-validation, procedural grid-margin, grid-scope diagnostic, OpenVLA audit, and embedded checklist evidence.",
                        "The rendered synthetic table checks the intended recovery-margin sampler over 30 paired seeds from `results/toy_30seed_v1/summary.csv`.",
                        "The rendered active-estimator validation checks that boundary-focused probes recover useful critical radii at a small fixed rollout budget over 30 paired seeds from `results/estimator_pair_30seed_v1/summary.csv`.",
                        "The procedural grid-margin section reports the completed 30-seed full-baseline confirmation.",
                        "The robot-suffix simulator reports a 30-seed coverage-aware BGR-Suffix variant.",
                        "The 30-seed suffix stress confirmation checks teacher quality, clutter, feasibility, and boundary sharpness.",
                        "BGR-Coverage beats uniform with 30/0 paired wins on clean success, final object RAUC, transfer RAUC, and AULC in all four stress cases.",
                        "The OpenVLA-OFT bridge includes corrected clean-mix diagnostics.",
                        "The packaged action-label/TFDS plumbing audit validates 2,048-transition matched BGR/random exports with 7D actions and 8D state.",
                        "OpenVLA therefore remains an audit of recovery curves, matched action/TFDS construction, and OpenVLA-OFT plumbing rather than a robotics success claim.",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                check_paper_readme_submission_framing(root),
                [f"{readme}: paper README submission framing ok"],
            )

    def test_paper_readme_submission_framing_rejects_draft_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text("The current main.tex is an anonymous AAAI submission draft.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "stale paper README submission framing"):
                check_paper_readme_submission_framing(root)

    def test_paper_readme_submission_framing_rejects_current_manuscript_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "The current `main.tex` is an anonymous AAAI submission manuscript.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale paper README submission framing"):
                check_paper_readme_submission_framing(root)

    def test_paper_readme_submission_framing_rejects_revision_history_now_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "The procedural grid-margin section now promotes the completed 30-seed full-baseline confirmation.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale paper README submission framing"):
                check_paper_readme_submission_framing(root)

    def test_paper_readme_submission_framing_rejects_ambiguous_diagnostic_tier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            readme = root / "paper" / "README.md"
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(
                "It has synthetic mechanism, estimator-validation, procedural, diagnostic, OpenVLA audit, and embedded checklist evidence.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale paper README submission framing"):
                check_paper_readme_submission_framing(root)

    def test_manuscript_framing_rejects_stale_suffix_overclaim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "In synthetic, procedural grid-margin, and robot-suffix simulator benchmarks, "
                "BGR improves recovery AUC, clean success, and learning-curve area over uniform replay.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale manuscript framing"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_checked_in_artifact_overclaim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "All experiments are driven by checked-in YAML configs and scripts.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale manuscript framing"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_stale_estimator_caption(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "Active BGR improves boundary-hit rate over uniform and coarse probing, "
                "while keeping critical-radius error close to the coarse sweep.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale manuscript framing"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_collapsed_evidence_tiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "The main claim is supported by controlled synthetic, procedural grid-margin, and robot-suffix experiments.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "main claim is supported"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_current_strongest_evidence_wording(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "The current strongest evidence is controlled procedural margin expansion.",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale manuscript framing"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_buried_proposition_statement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "\n".join(
                    [
                        "local margin-update model.",
                        r"\textbf{Proposition 1.} Fix a replay state.",
                        "In a robot-suffix simulator, a coverage-aware BGR variant improves final object recovery AUC",
                        "coverage-aware boundary replay expand recovery margins",
                        "Instantiations of the BGR interface",
                        "The first three rows are training experiments",
                        "OpenVLA row is an audit of recovery curves and data plumbing, not a fine-tuning claim",
                        "All experiments are driven by artifact YAML configs and scripts",
                        "Slurm launch commands, run ledger files",
                        "Prioritized experience replay ranks transitions by temporal-difference error",
                        "BGR is not an adversarial-training objective",
                        "active learning selects informative labels under a query budget",
                        "The novelty claim is narrow and falsifiable",
                        "If state-priority replay with uniform radii or loss-priority replay matches BGR",
                        "The radius-level ablation is therefore a negative control",
                        "state priority is held fixed and only the radius rule changes",
                        "changes the target from a level or transition score to an estimated curve crossing",
                        "and the replay action from revisiting a hard state to choosing a radius near that crossing",
                        "We use three evidence tiers",
                        "The main claim is supported by controlled synthetic and procedural grid-margin experiments",
                        "The robot-suffix simulator is a manipulation-style extension",
                        "four stress regimes",
                        "A four-condition 30-seed suffix stress sweep varies teacher quality",
                        "while still trailing uniform on median critical radius",
                        "Five-seed exploratory variants are reported only as exploratory evidence",
                        "OpenVLA/LIBERO results are learned-policy audits and infrastructure checks, not promoted positive claims",
                        "OpenVLA is a learned-policy audit rather than a robotics training claim",
                        "action-label/TFDS plumbing validates 2,048-transition matched BGR/random exports with 7D actions and 8D state",
                        "matched action/TFDS construction",
                        "Geometric intuition for BGR",
                        "inset showing BGR/uniform critical-radius distributions",
                        "We treat exact paired sign tests as consistency checks over shared seeds, not as substitutes for effect size",
                        "The following local calculation is a design rationale for the radius sampler and for the radius-level ablation",
                        "Local boundary intuition",
                        "BGR depends on a feasibility witness",
                        "synthetic and grid-margin benchmarks are constructed to expose recovery curves",
                        "A canonical Gym FrozenLake8x8-v1 diagnostic reinforces that limitation",
                        "RAUC and AULC are useful for measuring curve expansion, but they are author-defined integrals",
                        "300-step image-augmentation continuation",
                        "1,000-step low-learning-rate continuation",
                        "measure learned-policy brittleness and build matched fine-tuning datasets",
                        "current adaptation does not establish stable gains over the official checkpoint",
                        "BGR converts recovery-margin measurement into a replay curriculum",
                        "current learned-policy evidence remains an audit, not the final robotics result",
                        "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Proposition 1"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_accepts_evidence_tiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "\n".join(
                    [
                        "local margin-update model.",
                        "",
                        r"\textbf{Local Boundary Intuition.} Fix a replay state.",
                        "Synthetic and robot-suffix results are smaller and are reported as scoped support",
                        "In a robot-suffix simulator, a coverage-aware BGR variant improves final object recovery AUC",
                        "coverage-aware boundary replay expand recovery margins",
                        "Instantiations of the BGR interface",
                        "The first three rows are training experiments",
                        "OpenVLA row is an audit of recovery curves and data plumbing, not a fine-tuning claim",
                        "All experiments are driven by artifact YAML configs and scripts",
                        "Slurm launch commands, run ledger files",
                        "Prioritized experience replay ranks transitions by temporal-difference error",
                        "BGR is not an adversarial-training objective",
                        "active learning selects informative labels under a query budget",
                        "The novelty claim is narrow and falsifiable",
                        "If state-priority replay with uniform radii or loss-priority replay matches BGR",
                        "The radius-level ablation is therefore a negative control",
                        "state priority is held fixed and only the radius rule changes",
                        "changes the target from a level or transition score to an estimated curve crossing",
                        "and the replay action from revisiting a hard state to choosing a radius near that crossing",
                        "We use three evidence tiers",
                        "Evidence contract used throughout the experiments",
                        "independent-benchmark and learned-policy wins remain open requirements rather than hidden claims",
                        "The main claim is supported by controlled synthetic and procedural grid-margin experiments",
                        "The robot-suffix simulator is a manipulation-style extension",
                        "Gym-style standard-environment probes and OpenVLA/LIBERO results are scope checks and learned-policy audits, not promoted positive claims",
                        "not evidence that boundary-only replay is robust",
                        "promoted only after 30-seed confirmation, held-out replication, and stress sweeps",
                        "four stress regimes",
                        "A four-condition 30-seed suffix stress sweep varies teacher quality",
                        "while still trailing uniform on median critical radius",
                        "Five-seed exploratory variants are reported only as exploratory evidence",
                        "Five-seed exploratory variants and small pre-promotion standard-environment probes are reported only as scope diagnostics",
                        "Additional 4-seed pre-promotion probes also stay negative",
                        "OpenVLA/LIBERO results are learned-policy audits and infrastructure checks, not promoted positive claims",
                        "OpenVLA is a learned-policy audit rather than a robotics training claim",
                        "action-label/TFDS plumbing validates 2,048-transition matched BGR/random exports with 7D actions and 8D state",
                        "matched action/TFDS construction",
                        "Geometric intuition for BGR",
                        "final recovery curves for BGR, uniform, failure-only, fixed-radius, and PLR-loss replay",
                        "inset showing BGR/uniform critical-radius distributions",
                        "We treat exact paired sign tests as consistency checks over shared seeds, not as substitutes for effect size",
                        "The following local calculation is a design rationale for the radius sampler and for the radius-level ablation",
                        "not a convergence result, global robustness theorem, or margin-expansion guarantee",
                        "Local boundary intuition",
                        "Finite-grid estimator guarantee",
                        "Hoeffding plus a union bound",
                        "critical radius is within",
                        "This certifies boundary localization, not learner improvement",
                        "BGR depends on a feasibility witness",
                        "This is a real interface assumption, not free supervision or a learned success oracle",
                        "When no reliable witness is available, BGR should be treated as an audit tool or not applied",
                        "We only promote settings where the witness is exact or stress-tested",
                        "synthetic and grid-margin benchmarks are constructed to expose recovery curves",
                        "A canonical Gym FrozenLake8x8-v1 diagnostic reinforces that limitation",
                        "Standard-environment scope audits",
                        "RAUC and AULC are useful for measuring curve expansion, but they are author-defined integrals",
                        "300-step image-augmentation continuation",
                        "1,000-step low-learning-rate continuation",
                        "measure learned-policy brittleness and build matched fine-tuning datasets",
                        "current adaptation does not establish stable gains over the official checkpoint",
                        "BGR converts recovery-margin measurement into a replay curriculum",
                        "current learned-policy evidence remains an audit, not the final robotics result",
                        "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(check_manuscript_framing(paper), [f"{paper}: manuscript evidence-tier framing ok"])

    def test_manuscript_framing_rejects_weakened_suffix_rescue_scope(self):
        source_paper = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
        manuscript = source_paper.read_text(encoding="utf-8")
        scoped = "not evidence that boundary-only replay is robust"
        self.assertIn(scoped, manuscript)

        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(manuscript.replace(scoped, "evidence that boundary-only replay is robust"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "boundary-only replay is robust"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_estimator_guarantee_without_scope_caveat(self):
        source_paper = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
        manuscript = source_paper.read_text(encoding="utf-8")
        scoped = "This certifies boundary localization, not learner improvement"
        self.assertIn(scoped, manuscript)

        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(manuscript.replace(scoped, "This certifies boundary localization"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "This certifies boundary localization"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_weakened_radius_ablation_control(self):
        source_paper = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
        manuscript = source_paper.read_text(encoding="utf-8")
        scoped = "state priority is held fixed and only the radius rule changes"
        self.assertIn(scoped, manuscript)

        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(manuscript.replace(scoped, "state priority and radius sampling both change"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "state priority is held fixed"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_weakened_witness_scope(self):
        source_paper = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
        manuscript = source_paper.read_text(encoding="utf-8")
        scoped = "When no reliable witness is available, BGR should be treated as an audit tool or not applied"
        self.assertIn(scoped, manuscript)

        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(manuscript.replace(scoped, "BGR can infer feasibility during replay"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "When no reliable witness is available"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_boundary_figure_without_r80_distribution_inset(self):
        source_paper = Path(__file__).resolve().parents[1] / "paper" / "main.tex"
        manuscript = source_paper.read_text(encoding="utf-8")
        scoped = "inset showing BGR/uniform critical-radius distributions"
        self.assertIn(scoped, manuscript)

        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(manuscript.replace(scoped, "inset showing method diagnostics"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "critical-radius distributions"):
                check_manuscript_framing(paper)

    def test_manuscript_framing_rejects_remote_log_path_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text("All experiments are driven by artifact YAML configs and scripts. remote log paths\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "remote log paths"):
                check_manuscript_framing(paper)

    def test_abstract_compliance_accepts_short_anonymous_abstract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                r"\begin{abstract}Boundary replay estimates recovery curves without naming authors.\end{abstract}",
                encoding="utf-8",
            )

            self.assertEqual(check_abstract_compliance(paper), [f"{paper}: abstract compliance ok (8/350 words)"])

    def test_abstract_compliance_rejects_missing_abstract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(r"\section{Introduction}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing LaTeX abstract environment"):
                check_abstract_compliance(paper)

    def test_abstract_compliance_rejects_citations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(r"\begin{abstract}Boundary replay follows prior work \cite{example}.\end{abstract}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "abstract contains reference command"):
                check_abstract_compliance(paper)

    def test_abstract_compliance_rejects_overlength_abstract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(r"\begin{abstract}" + " ".join(["word"] * 351) + r"\end{abstract}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "abstract has 351 words"):
                check_abstract_compliance(paper)

    def test_abstract_compliance_rejects_double_blind_leaks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                r"\begin{abstract}See " + self.author_pattern("/" + "Users" + "/") + r"local build.\end{abstract}",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "double-blind leak pattern"):
                check_abstract_compliance(paper)

    def test_title_block_compliance_accepts_anonymous_title_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "\n".join(
                    [
                        r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                        r"\author{Anonymous Submission}",
                        r"\affiliations{}",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertEqual(check_title_block_compliance(paper), [f"{paper}: anonymous title block ok"])

    def test_title_block_compliance_rejects_missing_author_macro(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "\n".join(
                    [
                        r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                        r"\affiliations{}",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"missing LaTeX \\author macro"):
                check_title_block_compliance(paper)

    def test_title_block_compliance_rejects_named_author(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "\n".join(
                    [
                        r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                        r"\author{Named Author}",
                        r"\affiliations{}",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Anonymous Submission"):
                check_title_block_compliance(paper)

    def test_title_block_compliance_rejects_nonempty_affiliations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "\n".join(
                    [
                        r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                        r"\author{Anonymous Submission}",
                        r"\affiliations{Paper under double-blind review}",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "affiliations must be empty"):
                check_title_block_compliance(paper)

    def test_title_block_compliance_rejects_stale_title_term(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "\n".join(
                    [
                        r"\title{Bifurcation-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                        r"\author{Anonymous Submission}",
                        r"\affiliations{}",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stale title term"):
                check_title_block_compliance(paper)

    def test_title_block_compliance_rejects_double_blind_leaks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paper = Path(temp_dir) / "main.tex"
            paper.write_text(
                "\n".join(
                    [
                        r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                        r"\author{Anonymous Submission}",
                        r"\affiliations{" + self.author_pattern("/" + "Users" + "/") + r"local}",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "double-blind leak pattern"):
                check_title_block_compliance(paper)

    def test_rendered_title_block_sync_accepts_current_title_and_anonymous_author(self):
        source_text = "\n".join(
            [
                r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                r"\author{Anonymous Submission}",
                r"\affiliations{}",
            ]
        )
        rendered = "\n".join(
            [
                "Boundary-Guided Replay: A Mechanism Study of Success-Failure Boundary",
                "Learning",
                "Anonymous submission",
                "Abstract",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=source_text), mock.patch(
            "scripts.check_submission_package.pdf_page_text",
            return_value=rendered,
        ):
            self.assertEqual(
                check_rendered_title_block_sync(Path("paper/main.tex"), Path("paper/main.pdf")),
                ["paper/main.pdf: rendered/source title block sync ok"],
            )

    def test_rendered_title_block_sync_rejects_stale_rendered_title(self):
        source_text = "\n".join(
            [
                r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                r"\author{Anonymous Submission}",
                r"\affiliations{}",
            ]
        )
        rendered = "Bifurcation-Guided Replay: A Mechanism Study of Success-Failure Boundary Learning\nAnonymous submission"
        with mock.patch.object(Path, "read_text", return_value=source_text), mock.patch(
            "scripts.check_submission_package.pdf_page_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "does not match source title"):
                check_rendered_title_block_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_title_block_sync_rejects_missing_anonymous_author(self):
        source_text = "\n".join(
            [
                r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                r"\author{Anonymous Submission}",
                r"\affiliations{}",
            ]
        )
        rendered = "Boundary-Guided Replay: A Mechanism Study of Success-Failure Boundary Learning\nNamed Author"
        with mock.patch.object(Path, "read_text", return_value=source_text), mock.patch(
            "scripts.check_submission_package.pdf_page_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "missing anonymous author"):
                check_rendered_title_block_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_title_block_sync_rejects_stale_affiliation_placeholder(self):
        source_text = "\n".join(
            [
                r"\title{Boundary-Guided Replay: A Mechanism Study of Success--Failure Boundary Learning}",
                r"\author{Anonymous Submission}",
                r"\affiliations{}",
            ]
        )
        rendered = "\n".join(
            [
                "Boundary-Guided Replay: A Mechanism Study of Success-Failure Boundary Learning",
                "Anonymous submission",
                "Paper under double-blind review",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=source_text), mock.patch(
            "scripts.check_submission_package.pdf_page_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "stale rendered affiliation placeholder"):
                check_rendered_title_block_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_abstract_sync_accepts_matching_first_page_abstract(self):
        source_text = r"\begin{abstract}Boundary-focused probes estimate recovery curves under small fixed budgets.\end{abstract}"
        rendered = "\n".join(
            [
                "Boundary-Guided Replay",
                "Anonymous submission",
                "Abstract",
                "Boundary-focused probes estimate recovery curves under small fixed budgets.",
                "1 Introduction",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=source_text), mock.patch(
            "scripts.check_submission_package.pdf_page_text",
            return_value=rendered,
        ):
            self.assertEqual(
                check_rendered_abstract_sync(Path("paper/main.tex"), Path("paper/main.pdf")),
                ["paper/main.pdf: rendered/source abstract sync ok"],
            )

    def test_rendered_abstract_sync_rejects_stale_first_page_abstract(self):
        source_text = r"\begin{abstract}Boundary-focused probes estimate recovery curves under small fixed budgets.\end{abstract}"
        rendered = "Abstract\nA stale abstract discusses unrelated replay claims."
        with mock.patch.object(Path, "read_text", return_value=source_text), mock.patch(
            "scripts.check_submission_package.pdf_page_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "does not match source abstract"):
                check_rendered_abstract_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_pdf_metadata_parses_pdfinfo_output(self):
        output = "\n".join(
            [
                "Title:           main",
                "JavaScript:      no",
                "Pages:           6",
                "Page size:       612 x 792 pts (letter)",
                "File size:       225675 bytes",
            ]
        )
        with mock.patch("scripts.check_submission_package.run_text_command", return_value=output):
            metadata = pdf_metadata("paper/main.pdf")  # type: ignore[arg-type]

        self.assertEqual(metadata.pages, 6)
        self.assertEqual(metadata.page_size, "612 x 792 pts (letter)")
        self.assertEqual(metadata.file_size, 225675)
        self.assertEqual(metadata.info["Title"], "main")
        self.assertEqual(metadata.info["JavaScript"], "no")

    def test_pdf_font_rows_parses_pdffonts_output(self):
        output = "\n".join(
            [
                "name                                 type              encoding         emb sub uni object ID",
                "------------------------------------ ----------------- ---------------- --- --- --- ---------",
                "ABCDXY+NimbusRomNo9L-Regu            Type 1            Custom           yes yes yes     34  0",
                "EFGHXY+CMR10                         Type 1            Builtin          yes yes yes     38  0",
            ]
        )
        with mock.patch("scripts.check_submission_package.run_text_command", return_value=output):
            rows = pdf_font_rows(Path("paper/main.pdf"))

        self.assertEqual(rows[0]["name"], "ABCDXY+NimbusRomNo9L-Regu")
        self.assertEqual(rows[0]["type"], "Type 1")
        self.assertEqual(rows[0]["embedded"], "yes")
        self.assertEqual(rows[1]["encoding"], "Builtin")

    def test_pdf_font_hygiene_rejects_type3_fonts(self):
        output = "\n".join(
            [
                "name                                 type              encoding         emb sub uni object ID",
                "------------------------------------ ----------------- ---------------- --- --- --- ---------",
                "BADFONT+CMR10                        Type 3            Custom           yes yes yes     10  0",
            ]
        )
        with mock.patch("scripts.check_submission_package.run_text_command", return_value=output):
            with self.assertRaisesRegex(ValueError, "Type 3 PDF font"):
                check_pdf_font_hygiene("main PDF", Path("paper/main.pdf"))

    def test_pdf_font_hygiene_rejects_unembedded_fonts(self):
        output = "\n".join(
            [
                "name                                 type              encoding         emb sub uni object ID",
                "------------------------------------ ----------------- ---------------- --- --- --- ---------",
                "Times-Roman                          Type 1            WinAnsi          no  no  no      10  0",
            ]
        )
        with mock.patch("scripts.check_submission_package.run_text_command", return_value=output):
            with self.assertRaisesRegex(ValueError, "unembedded PDF font"):
                check_pdf_font_hygiene("main PDF", Path("paper/main.pdf"))

    def test_pdf_font_hygiene_accepts_embedded_type1_fonts(self):
        output = "\n".join(
            [
                "name                                 type              encoding         emb sub uni object ID",
                "------------------------------------ ----------------- ---------------- --- --- --- ---------",
                "ABCDXY+NimbusRomNo9L-Regu            Type 1            Custom           yes yes yes     34  0",
            ]
        )
        with mock.patch("scripts.check_submission_package.run_text_command", return_value=output):
            self.assertEqual(
                check_pdf_font_hygiene("main PDF", Path("paper/main.pdf")),
                ["main PDF: PDF font hygiene ok"],
            )

    def test_pdf_metadata_hygiene_accepts_anonymous_tex_metadata(self):
        metadata = PdfMetadata(
            pages=9,
            page_size="612 x 792 pts (letter)",
            file_size=263248,
            info={
                "Creator": "TeX",
                "Producer": "pdfTeX-1.40.22",
                "CreationDate": "Wed Jun  3 00:38:48 2026 PDT",
                "ModDate": "Wed Jun  3 00:38:48 2026 PDT",
                "Custom Metadata": "no",
                "Metadata Stream": "no",
                "Form": "none",
                "JavaScript": "no",
                "Encrypted": "no",
            },
        )

        self.assertEqual(check_pdf_metadata_hygiene("main PDF", metadata), ["main PDF: PDF metadata hygiene ok"])

    def test_pdf_metadata_hygiene_rejects_custom_metadata(self):
        metadata = PdfMetadata(
            pages=9,
            page_size="612 x 792 pts (letter)",
            file_size=263248,
            info={
                "Custom Metadata": "yes",
                "Metadata Stream": "no",
                "Form": "none",
                "JavaScript": "no",
                "Encrypted": "no",
            },
        )

        with self.assertRaisesRegex(ValueError, "expected cleared custom PDF metadata"):
            check_pdf_metadata_hygiene("main PDF", metadata)

    def test_pdf_metadata_hygiene_rejects_identity_fields(self):
        metadata = PdfMetadata(
            pages=9,
            page_size="612 x 792 pts (letter)",
            file_size=263248,
            info={
                "Author": "Anonymous Submission",
                "Form": "none",
                "JavaScript": "no",
                "Encrypted": "no",
            },
        )

        with self.assertRaisesRegex(ValueError, "identity-bearing PDF metadata field"):
            check_pdf_metadata_hygiene("main PDF", metadata)

    def test_pdf_metadata_hygiene_rejects_metadata_double_blind_leaks(self):
        metadata = PdfMetadata(
            pages=9,
            page_size="612 x 792 pts (letter)",
            file_size=263248,
            info={
                "Creator": self.author_pattern("/" + "Users" + "/") + "texbuild",
                "Custom Metadata": "no",
                "Metadata Stream": "no",
                "Form": "none",
                "JavaScript": "no",
                "Encrypted": "no",
            },
        )

        with self.assertRaisesRegex(ValueError, "double-blind leak pattern"):
            check_pdf_metadata_hygiene("main PDF", metadata)

    def test_pdf_metadata_hygiene_rejects_unsafe_pdf_features(self):
        metadata = PdfMetadata(
            pages=9,
            page_size="612 x 792 pts (letter)",
            file_size=263248,
            info={
                "Custom Metadata": "no",
                "Metadata Stream": "no",
                "Form": "AcroForm",
                "JavaScript": "yes",
                "Encrypted": "no",
            },
        )

        with self.assertRaisesRegex(ValueError, "unsafe PDF metadata feature"):
            check_pdf_metadata_hygiene("main PDF", metadata)

    def test_checklist_pdf_requires_section_order(self):
        metadata_output = "\n".join(
            [
                "Pages:           3",
                "Page size:       612 x 792 pts (letter)",
                "File size:       50000 bytes",
                "Custom Metadata: no",
                "Metadata Stream: no",
                "Encrypted:       no",
                "Form:            none",
                "JavaScript:      no",
            ]
        )
        font_output = "\n".join(
            [
                "name                                 type              encoding         emb sub uni object ID",
                "------------------------------------ ----------------- ---------------- --- --- --- ---------",
                "ABCDXY+NimbusRomNo9L-Regu            Type 1            Custom           yes yes yes     14  0",
            ]
        )
        out_of_order_text = "\n".join(
            [
                "Reproducibility Checklist",
                "3. Dataset Usage",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "synthetic training, estimator validation, grid-margin full-baseline, grid ablation, grid sensitivity confirmations, and robot-suffix coverage comparisons use 30 paired seeds",
                "grid learning-curve diagnostics use 15 seed pairs",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "4. Computational Experiments",
            ]
        )
        with mock.patch(
            "scripts.check_submission_package.run_text_command",
            side_effect=[metadata_output, font_output, out_of_order_text],
        ):
            with self.assertRaisesRegex(ValueError, "out of order"):
                check_checklist_pdf(Path("paper/ReproducibilityChecklist.pdf"))

    def test_checklist_pdf_rejects_checked_in_artifact_framing(self):
        metadata_output = "\n".join(
            [
                "Pages:           3",
                "Page size:       612 x 792 pts (letter)",
                "File size:       50000 bytes",
                "Custom Metadata: no",
                "Metadata Stream: no",
                "Encrypted:       no",
                "Form:            none",
                "JavaScript:      no",
            ]
        )
        font_output = "\n".join(
            [
                "name                                 type              encoding         emb sub uni object ID",
                "------------------------------------ ----------------- ---------------- --- --- --- ---------",
                "ABCDXY+NimbusRomNo9L-Regu            Type 1            Custom           yes yes yes     14  0",
            ]
        )
        stale_text = "\n".join(
            [
                "Reproducibility Checklist",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "generated by checked-in scripts/configs",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "synthetic training, estimator validation, grid-margin full-baseline, grid ablation, grid sensitivity confirmations, and robot-suffix coverage comparisons use 30 paired seeds",
                "grid learning-curve diagnostics use 15 seed pairs",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "3. Dataset Usage",
                "4. Computational Experiments",
            ]
        )
        with mock.patch(
            "scripts.check_submission_package.run_text_command",
            side_effect=[metadata_output, font_output, stale_text],
        ):
            with self.assertRaisesRegex(ValueError, "stale rendered checklist artifact framing"):
                check_checklist_pdf(Path("paper/ReproducibilityChecklist.pdf"))

    def test_checklist_pdf_rejects_stale_seed_framing(self):
        metadata_output = "\n".join(
            [
                "Pages:           3",
                "Page size:       612 x 792 pts (letter)",
                "File size:       50000 bytes",
                "Custom Metadata: no",
                "Metadata Stream: no",
                "Encrypted:       no",
                "Form:            none",
                "JavaScript:      no",
            ]
        )
        font_output = "\n".join(
            [
                "name                                 type              encoding         emb sub uni object ID",
                "------------------------------------ ----------------- ---------------- --- --- --- ---------",
                "ABCDXY+NimbusRomNo9L-Regu            Type 1            Custom           yes yes yes     14  0",
            ]
        )
        stale_text = "\n".join(
            [
                "Reproducibility Checklist",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "The main synthetic, grid-margin, suffix, and estimator comparisons use 15 paired seeds.",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "3. Dataset Usage",
                "4. Computational Experiments",
            ]
        )
        with mock.patch(
            "scripts.check_submission_package.run_text_command",
            side_effect=[metadata_output, font_output, stale_text],
        ):
            with self.assertRaisesRegex(ValueError, "stale rendered checklist claim framing"):
                check_checklist_pdf(Path("paper/ReproducibilityChecklist.pdf"))

    def test_checklist_pdf_rejects_answer_spacing_regression(self):
        metadata_output = "\n".join(
            [
                "Pages:           3",
                "Page size:       612 x 792 pts (letter)",
                "File size:       50000 bytes",
                "Custom Metadata: no",
                "Metadata Stream: no",
                "Encrypted:       no",
                "Form:            none",
                "JavaScript:      no",
            ]
        )
        font_output = "\n".join(
            [
                "name                                 type              encoding         emb sub uni object ID",
                "------------------------------------ ----------------- ---------------- --- --- --- ---------",
                "ABCDXY+NimbusRomNo9L-Regu            Type 1            Custom           yes yes yes     14  0",
            ]
        )
        stale_text = "\n".join(
            [
                "Reproducibility Checklist",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "1.1. Includes a conceptual outline (yes/no)yes.",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "synthetic training, estimator validation, grid-margin full-baseline, grid ablation, grid sensitivity confirmations, and robot-suffix coverage comparisons use 30 paired seeds",
                "grid learning-curve diagnostics use 15 seed pairs",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "3. Dataset Usage",
                "4. Computational Experiments",
            ]
        )
        with mock.patch(
            "scripts.check_submission_package.run_text_command",
            side_effect=[metadata_output, font_output, stale_text],
        ):
            with self.assertRaisesRegex(ValueError, "answer spacing regression"):
                check_checklist_pdf(Path("paper/ReproducibilityChecklist.pdf"))

    def test_embedded_checklist_must_follow_references(self):
        text = "\n".join(
            [
                "Boundary-Guided Replay",
                "Reproducibility Checklist",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "synthetic training, estimator validation, grid-margin full-baseline, grid ablation, grid sensitivity confirmations, and robot-suffix coverage comparisons use 30 paired seeds",
                "grid learning-curve diagnostics use 15 seed pairs",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "3. Dataset Usage",
                "4. Computational Experiments",
                "References",
            ]
        )

        with self.assertRaisesRegex(ValueError, "before references"):
            check_embedded_checklist_text(Path("paper/main.pdf"), text)

    def test_embedded_checklist_rejects_checked_in_artifact_framing(self):
        text = "\n".join(
            [
                "Boundary-Guided Replay",
                "References",
                "Reproducibility Checklist",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "Final configs and ablation variants are checked in",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "synthetic training, estimator validation, grid-margin full-baseline, grid ablation, grid sensitivity confirmations, and robot-suffix coverage comparisons use 30 paired seeds",
                "grid learning-curve diagnostics use 15 seed pairs",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "3. Dataset Usage",
                "4. Computational Experiments",
            ]
        )

        with self.assertRaisesRegex(ValueError, "stale rendered checklist artifact framing"):
            check_embedded_checklist_text(Path("paper/main.pdf"), text)

    def test_embedded_checklist_rejects_stale_seed_framing(self):
        text = "\n".join(
            [
                "Boundary-Guided Replay",
                "References",
                "Reproducibility Checklist",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "The main synthetic, grid-margin, suffix, and estimator comparisons use 15 paired seeds.",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "3. Dataset Usage",
                "4. Computational Experiments",
            ]
        )

        with self.assertRaisesRegex(ValueError, "stale rendered checklist claim framing"):
            check_embedded_checklist_text(Path("paper/main.pdf"), text)

    def test_embedded_checklist_rejects_answer_spacing_regression(self):
        text = "\n".join(
            [
                "Boundary-Guided Replay",
                "References",
                "Reproducibility Checklist",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "1.1. Includes a conceptual outline (yes/no)yes.",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "synthetic training, estimator validation, grid-margin full-baseline, grid ablation, grid sensitivity confirmations, and robot-suffix coverage comparisons use 30 paired seeds",
                "grid learning-curve diagnostics use 15 seed pairs",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "3. Dataset Usage",
                "4. Computational Experiments",
            ]
        )

        with self.assertRaisesRegex(ValueError, "answer spacing regression"):
            check_embedded_checklist_text(Path("paper/main.pdf"), text)

    def test_embedded_checklist_accepts_ordered_sections_after_references(self):
        text = "\n".join(
            [
                "Boundary-Guided Replay",
                "References",
                "Reproducibility Checklist",
                "current BGR submission package",
                "run ledgers are in the anonymous artifact",
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "synthetic training, estimator validation, grid-margin full-baseline, grid ablation, grid sensitivity confirmations, and robot-suffix coverage comparisons use 30 paired seeds",
                "grid learning-curve diagnostics use 15 seed pairs",
                "1. General Paper Structure",
                "2. Theoretical Contributions",
                "3. Dataset Usage",
                "4. Computational Experiments",
            ]
        )

        messages = check_embedded_checklist_text(Path("paper/main.pdf"), text)

        self.assertEqual(messages, ["paper/main.pdf: embedded checklist after references ok"])

    def test_checklist_source_framing_accepts_current_seed_split(self):
        text = "\n".join(
            [
                "training-loop method box",
                "method section defines replayable states and recovery curves",
                "states critical radii, BGR priority",
                "synthetic training, estimator validation, grid-margin full-baseline, grid ablation, grid sensitivity confirmations, and robot-suffix coverage comparisons use 30 paired seeds",
                "grid learning-curve diagnostics use 15 seed pairs",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=text):
            self.assertEqual(
                check_checklist_source_framing(Path("paper/ReproducibilityChecklist.tex")),
                ["paper/ReproducibilityChecklist.tex: checklist source claim framing ok"],
            )

    def test_checklist_source_framing_rejects_stale_seed_split(self):
        text = "\n".join(
            [
                "training-loop method box",
                "The main synthetic, grid-margin, suffix, and estimator comparisons use 15 paired seeds.",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(ValueError, "stale checklist source claim framing"):
                check_checklist_source_framing(Path("paper/ReproducibilityChecklist.tex"))

    def test_main_pdf_rejects_central_table_order_regression(self):
        rendered = " ".join(
            [
                "Boundary-Guided Replay",
                "coverage-aware BGR variant improves",
                "OpenVLA/LIBERO audits",
                "not promoted positive claims",
                "run ledger files",
                "30-seed synthetic, estimator, grid, and suffix",
                "Suffix RAUC vs clean-only",
                "Suffix RAUC vs loss-priority",
                "Suffix transfer vs uniform",
                "Suffix AULC vs uniform",
                "< 0.0001",
                "Method box: BGR training loop without domain-specific learner details.",
                "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                "Table 4: Selected paired effect sizes",
                "Thirty-seed procedural grid-margin full-baseline comparison",
                "Table 5: Probe-efficiency validation",
                "Grid-margin ablations over 30 seeds",
                "Robot Suffix Study",
                "Table 7: Grid-margin ablations over 30 seeds",
                "Table 7 isolates the replay decisions inside BGR",
            ]
        )
        with mock.patch("scripts.check_submission_package.assert_letter_pdf", return_value=["pdf ok"]), mock.patch(
            "scripts.check_submission_package.check_technical_page_limit",
            return_value=["page ok"],
        ), mock.patch("scripts.check_submission_package.pdf_text", return_value=rendered), mock.patch(
            "scripts.check_submission_package.check_embedded_checklist_text",
            return_value=["checklist ok"],
        ), mock.patch(
            "scripts.check_submission_package.pdf_layout_text",
            return_value=(
                "Grid vs uniform final RAUC 30 0.0381 "
                "Suffix RAUC vs clean-only final RAUC 30 0.2338"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "rendered grid tables crossed into suffix section"):
                check_main_pdf(Path("paper/main.pdf"))

    def test_main_pdf_accepts_grid_table_order(self):
        rendered = " ".join(
            [
                "Boundary-Guided Replay",
                "coverage-aware BGR variant improves",
                "OpenVLA/LIBERO audits",
                "not promoted positive claims",
                "run ledger files",
                "30-seed synthetic, estimator, grid, and suffix",
                "Suffix RAUC vs clean-only",
                "Suffix RAUC vs loss-priority",
                "Suffix transfer vs uniform",
                "Suffix AULC vs uniform",
                "< 0.0001",
                "Method box: BGR training loop without domain-specific learner details.",
                "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                "Table 4: Selected paired effect sizes",
                "Table 5: Probe-efficiency validation",
                "Thirty-seed procedural grid-margin full-baseline comparison",
                "Grid-margin ablations over 30 seeds",
                "Table 7: Grid-margin ablations over 30 seeds",
                "Table 7 isolates the replay decisions inside BGR",
                "Robot Suffix Study",
            ]
        )
        with mock.patch("scripts.check_submission_package.assert_letter_pdf", return_value=["pdf ok"]), mock.patch(
            "scripts.check_submission_package.check_technical_page_limit",
            return_value=["page ok"],
        ), mock.patch("scripts.check_submission_package.pdf_text", return_value=rendered), mock.patch(
            "scripts.check_submission_package.check_embedded_checklist_text",
            return_value=["checklist ok"],
        ), mock.patch(
            "scripts.check_submission_package.pdf_layout_text",
            return_value=(
                "Grid vs uniform final RAUC 30 0.0381 "
                "Suffix RAUC vs clean-only final RAUC 30 0.2338"
            ),
        ):
            self.assertEqual(
                check_main_pdf(Path("paper/main.pdf")),
                ["pdf ok", "page ok", "checklist ok", "paper/main.pdf: rendered title/framing ok"],
            )

    def test_main_pdf_rejects_stale_grid_significance_row(self):
        rendered = " ".join(
            [
                "Boundary-Guided Replay",
                "coverage-aware BGR variant improves",
                "OpenVLA/LIBERO audits",
                "not promoted positive claims",
                "run ledger files",
                "30-seed synthetic, estimator, grid, and suffix",
                "Suffix RAUC vs clean-only",
                "Suffix RAUC vs loss-priority",
                "Suffix transfer vs uniform",
                "Suffix AULC vs uniform",
                "< 0.0001",
                "Method box: BGR training loop without domain-specific learner details.",
                "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                "Table 4: Selected paired effect sizes",
                "Table 5: Probe-efficiency validation",
                "Thirty-seed procedural grid-margin full-baseline comparison",
                "Grid-margin ablations over 30 seeds",
                "Table 7: Grid-margin ablations over 30 seeds",
                "Table 7 isolates the replay decisions inside BGR",
                "Robot Suffix Study",
            ]
        )
        with mock.patch("scripts.check_submission_package.assert_letter_pdf", return_value=["pdf ok"]), mock.patch(
            "scripts.check_submission_package.check_technical_page_limit",
            return_value=["page ok"],
        ), mock.patch("scripts.check_submission_package.pdf_text", return_value=rendered), mock.patch(
            "scripts.check_submission_package.check_embedded_checklist_text",
            return_value=["checklist ok"],
        ), mock.patch(
            "scripts.check_submission_package.pdf_layout_text",
            return_value=(
                "Grid vs uniform final RAUC 30 0.0381 "
                "Suffix RAUC vs clean-only final RAUC 30 0.2338 "
                "Grid vs uniform final RAUC 15 0.0084"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "stale rendered significance table row"):
                check_main_pdf(Path("paper/main.pdf"))

    def test_main_pdf_rejects_orphaned_layout_word_fragment(self):
        rendered = " ".join(
            [
                "Boundary-Guided Replay",
                "coverage-aware BGR variant improves",
                "OpenVLA/LIBERO audits",
                "not promoted positive claims",
                "run ledger files",
                "30-seed synthetic, estimator, grid, and suffix",
                "Suffix RAUC vs clean-only",
                "Suffix RAUC vs loss-priority",
                "Suffix transfer vs uniform",
                "Suffix AULC vs uniform",
                "< 0.0001",
                "Method box: BGR training loop without domain-specific learner details.",
                "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                "Table 4: Selected paired effect sizes",
                "Table 5: Probe-efficiency validation",
                "Thirty-seed procedural grid-margin full-baseline comparison",
                "Table 7: Grid-margin ablations over 30 seeds",
                "Robot Suffix Study",
            ]
        )
        with mock.patch("scripts.check_submission_package.assert_letter_pdf", return_value=["pdf ok"]), mock.patch(
            "scripts.check_submission_package.check_technical_page_limit",
            return_value=["page ok"],
        ), mock.patch("scripts.check_submission_package.pdf_text", return_value=rendered), mock.patch(
            "scripts.check_submission_package.check_embedded_checklist_text",
            return_value=["checklist ok"],
        ), mock.patch(
            "scripts.check_submission_package.pdf_layout_text",
            return_value=(
                "Grid vs uniform final RAUC 30 0.0381 "
                "Suffix RAUC vs clean-only final RAUC 30 0.2338 "
                "mative."
            ),
        ):
            with self.assertRaisesRegex(ValueError, "orphaned rendered word fragment"):
                check_main_pdf(Path("paper/main.pdf"))

    def test_main_pdf_rejects_stale_artifact_scope_wording(self):
        rendered = " ".join(
            [
                "Boundary-Guided Replay",
                "coverage-aware BGR variant improves",
                "OpenVLA/LIBERO audits",
                "not promoted positive claims",
                "remote log paths",
                "run ledger files",
                "30-seed synthetic, estimator, grid, and suffix",
                "Suffix RAUC vs clean-only",
                "Suffix RAUC vs loss-priority",
                "Suffix transfer vs uniform",
                "Suffix AULC vs uniform",
                "< 0.0001",
                "Method box: BGR training loop without domain-specific learner details.",
                "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                "Table 4: Selected paired effect sizes",
                "Table 5: Probe-efficiency validation",
                "Thirty-seed procedural grid-margin full-baseline comparison",
                "Table 7: Grid-margin ablations over 30 seeds",
                "Robot Suffix Study",
            ]
        )
        with mock.patch("scripts.check_submission_package.assert_letter_pdf", return_value=["pdf ok"]), mock.patch(
            "scripts.check_submission_package.check_technical_page_limit",
            return_value=["page ok"],
        ), mock.patch("scripts.check_submission_package.pdf_text", return_value=rendered):
            with self.assertRaisesRegex(ValueError, "stale rendered artifact-scope wording"):
                check_main_pdf(Path("paper/main.pdf"))

    def test_main_pdf_rejects_stale_method_box_caption(self):
        rendered = " ".join(
            [
                "Boundary-Guided Replay",
                "coverage-aware BGR variant improves",
                "OpenVLA/LIBERO audits",
                "not promoted positive claims",
                "run ledger files",
                "30-seed synthetic, estimator, grid, and suffix",
                "Suffix RAUC vs clean-only",
                "Suffix RAUC vs loss-priority",
                "Suffix transfer vs uniform",
                "Suffix AULC vs uniform",
                "< 0.0001",
                "Figure 1: BGR training loop without domain-specific learner details",
                "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                "Table 4: Selected paired effect sizes",
                "Table 5: Probe-efficiency validation",
                "Thirty-seed procedural grid-margin full-baseline comparison",
                "Table 7: Grid-margin ablations over 30 seeds",
                "Robot Suffix Study",
            ]
        )
        with mock.patch("scripts.check_submission_package.assert_letter_pdf", return_value=["pdf ok"]), mock.patch(
            "scripts.check_submission_package.check_technical_page_limit",
            return_value=["page ok"],
        ), mock.patch("scripts.check_submission_package.pdf_text", return_value=rendered):
            with self.assertRaisesRegex(ValueError, "stale rendered method-box caption"):
                check_main_pdf(Path("paper/main.pdf"))

    def test_main_pdf_rejects_stale_estimator_caption(self):
        rendered = " ".join(
            [
                "Boundary-Guided Replay",
                "coverage-aware BGR variant improves",
                "OpenVLA/LIBERO audits",
                "not promoted positive claims",
                "run ledger files",
                "30-seed synthetic, estimator, grid, and suffix",
                "Suffix RAUC vs clean-only",
                "Suffix RAUC vs loss-priority",
                "Suffix transfer vs uniform",
                "Suffix AULC vs uniform",
                "< 0.0001",
                "Method box: BGR training loop without domain-specific learner details.",
                "Active BGR improves boundary-hit rate over uniform and coarse probing",
                "Table 4: Selected paired effect sizes",
                "Table 5: Probe-efficiency validation",
                "Thirty-seed procedural grid-margin full-baseline comparison",
                "Table 7: Grid-margin ablations over 30 seeds",
                "Robot Suffix Study",
            ]
        )
        with mock.patch("scripts.check_submission_package.assert_letter_pdf", return_value=["pdf ok"]), mock.patch(
            "scripts.check_submission_package.check_technical_page_limit",
            return_value=["page ok"],
        ), mock.patch("scripts.check_submission_package.pdf_text", return_value=rendered):
            with self.assertRaisesRegex(ValueError, "stale rendered estimator-caption wording"):
                check_main_pdf(Path("paper/main.pdf"))

    def test_main_pdf_rejects_stale_current_strongest_evidence_wording(self):
        rendered = " ".join(
            [
                "Boundary-Guided Replay",
                "coverage-aware BGR variant improves",
                "OpenVLA/LIBERO audits",
                "not promoted positive claims",
                "run ledger files",
                "30-seed synthetic, estimator, grid, and suffix",
                "Suffix RAUC vs clean-only",
                "Suffix RAUC vs loss-priority",
                "Suffix transfer vs uniform",
                "Suffix AULC vs uniform",
                "< 0.0001",
                "Method box: BGR training loop without domain-specific learner details.",
                "Active BGR improves boundary-hit rate over uniform probing at the same rollout budget",
                "The current strongest evidence is controlled procedural margin expansion",
                "Table 4: Selected paired effect sizes",
                "Table 5: Probe-efficiency validation",
                "Thirty-seed procedural grid-margin full-baseline comparison",
                "Table 7: Grid-margin ablations over 30 seeds",
                "Robot Suffix Study",
            ]
        )
        with mock.patch("scripts.check_submission_package.assert_letter_pdf", return_value=["pdf ok"]), mock.patch(
            "scripts.check_submission_package.check_technical_page_limit",
            return_value=["page ok"],
        ), mock.patch("scripts.check_submission_package.pdf_text", return_value=rendered):
            with self.assertRaisesRegex(ValueError, "stale rendered limitations wording"):
                check_main_pdf(Path("paper/main.pdf"))

    def test_rendered_source_sync_rejects_stale_openvla_pdf(self):
        source = "In the p1024 diagnostic, BGR reports 0.8333 vs. 0.8000."
        rendered = "In the latest p512 clean-mix diagnostic, BGR reports 0.8167 vs. 0.8000."
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "stale rendered OpenVLA"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_rejects_latest_p1024_pdf_wording(self):
        source = "In the p1024 diagnostic, BGR reports 0.8333 vs. 0.8000."
        rendered = "In the latest p1024 diagnostic, BGR reports 0.8333 vs. 0.8000."
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "stale rendered OpenVLA"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_rejects_bgrselected_pdf_wording(self):
        source = "In the p1024 diagnostic, BGR and matched random both score 14/15 clean episodes."
        rendered = "In the p1024 clean-mix diagnostic, BGRselected and matched-random adaptation score 14/15 clean episodes."
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "stale rendered OpenVLA"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_rejects_reference_overlap_markers(self):
        source = "In the p1024 diagnostic, BGR reports 0.8333 vs. 0.8000."
        rendered = "same eval. In the p1024 clean-mix diagnostic, BGR and matchedChevalier-Boisvert"
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "rendered layout overlap"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_requires_artifact_scope_wording(self):
        source = "All experiments are driven by artifact YAML configs and scripts. run ledger files"
        rendered = "All experiments are driven by artifact YAML configs and scripts. remote log paths"
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "stale rendered artifact-scope wording"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_requires_suffix_replication_pdf_text(self):
        source = "A held-out seeds 30--59 replication gives 0.4972 vs. 0.4859 RAUC."
        rendered = "A 30-seed coverage-aware run fixes high-radius undercoverage."
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "missing rendered suffix replication"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_accepts_suffix_replication_pdf_text(self):
        source = "A held-out seeds 30--59 replication gives 0.4972 vs. 0.4859 RAUC."
        rendered = "A held-out seeds 30-59 replication gives 0.4972 vs. 0.4859 RAUC, also with 30/0 wins."
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            messages = check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

        self.assertEqual(messages, ["paper/main.pdf: rendered/source OpenVLA sync guard not triggered"])

    def test_rendered_source_sync_requires_grid_replication_pdf_text(self):
        source = "A held-out seeds 30--59 BGR-vs-uniform replication gives 0.4340 vs. 0.3967 RAUC."
        rendered = "In a completed 30-seed paired grid-margin full-baseline run."
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "missing rendered grid replication"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_accepts_grid_replication_pdf_text(self):
        source = "A held-out seeds 30--59 BGR-vs-uniform replication gives 0.4340 vs. 0.3967 RAUC."
        rendered = "A held-out seeds 30-59 BGR-vs-uniform replication gives 0.4340 vs. 0.3967 RAUC, also with 30/0 wins."
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            messages = check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

        self.assertEqual(messages, ["paper/main.pdf: rendered/source OpenVLA sync guard not triggered"])

    def test_rendered_source_sync_requires_frozenlake_limitation_pdf_text(self):
        source = "Standard-environment scope audits. FrozenLake8x8 & BGR 0.5453 & signs 14/16, 13/17."
        rendered = "The paper does not claim a clear win on an independent pre-existing robustness benchmark."
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "missing rendered scope-audit"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_accepts_frozenlake_limitation_pdf_text(self):
        source = "Standard-environment scope audits. FrozenLake8x8 & BGR 0.5453 & signs 14/16, 13/17."
        rendered = (
            "Standard-environment scope audits. FrozenLake8x8v1 BGR 0.5453 "
            "signs 14/16, 13/17. MiniGrid FourRooms BGR-Coverage 0.6077 "
            "failureonly 0.7940 fixed 0.7587. DoorKey-6x6 BGR-Coverage 0.4846 "
            "uniform 0.6384. LavaCrossingS9N3 BGR-Coverage 0.3547 uniform 0.4165. "
            "PointMaze UMaze BGR-Clean-Shield 0.2448 failureonly 0.5458 "
            "2/4 wins r20 0.1167 vs. 0.2500."
        )
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            messages = check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

        self.assertEqual(messages, ["paper/main.pdf: rendered/source OpenVLA sync guard not triggered"])

    def test_rendered_source_sync_accepts_current_openvla_pdf(self):
        source = "In the p1024 diagnostic, BGR reports 0.8333 vs. 0.8000."
        rendered = "p1024 diagnostic success 0.8333 vs. 0.8000 at official 0.8167; pooled 0.8550 vs. 0.8400 at official 0.8700"
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            messages = check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

        self.assertEqual(messages, ["paper/main.pdf: rendered/source OpenVLA p1024 sync ok"])

    def test_rendered_source_sync_requires_p2048_when_source_mentions_it(self):
        source = "In the p1024 diagnostic, BGR reports 0.8333 vs. 0.8000. A p2048 scale-up remains diagnostic."
        rendered = "p1024 diagnostic success 0.8333 vs. 0.8000 at official 0.8167; pooled 0.8550 vs. 0.8400 at official 0.8700"
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            with self.assertRaisesRegex(ValueError, "missing rendered OpenVLA"):
                check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

    def test_rendered_source_sync_accepts_p2048_openvla_pdf(self):
        source = "In the p1024 diagnostic, BGR reports 0.8333 vs. 0.8000. A p2048 scale-up remains diagnostic."
        rendered = (
            "p1024 diagnostic success 0.8333 vs. 0.8000 at official 0.8167; "
            "pooled 0.8550 vs. 0.8400 at official 0.8700; "
            "p2048 full-goal audit 99/100 clean and 367/400 vs. 368/400 perturbed"
        )
        with mock.patch.object(Path, "read_text", return_value=source), mock.patch(
            "scripts.check_submission_package.pdf_text",
            return_value=rendered,
        ):
            messages = check_rendered_source_sync(Path("paper/main.tex"), Path("paper/main.pdf"))

        self.assertEqual(messages, ["paper/main.pdf: rendered/source OpenVLA p1024/p2048 sync ok"])

    def test_author_kit_source_rejects_forbidden_package_and_page_break(self):
        with self.subTest("package"):
            path = Path("main.tex")
            with mock.patch.object(Path, "read_text", return_value=r"\usepackage{hyperref}"):
                with self.assertRaisesRegex(ValueError, "forbidden AAAI package"):
                    check_author_kit_source(path)
        with self.subTest("caption package"):
            text = "\n".join(
                [
                    r"\usepackage[submission]{AuthorKit27/aaai2027}",
                    r"\usepackage{caption}",
                    r"\noindent\textit{Method box: BGR training loop without domain-specific learner details.}",
                ]
            )
            with mock.patch.object(Path, "read_text", return_value=text):
                with self.assertRaisesRegex(ValueError, "forbidden AAAI package"):
                    check_author_kit_source(path)
        with self.subTest("page break"):
            path = Path("main.tex")
            with mock.patch.object(Path, "read_text", return_value=r"\clearpage"):
                with self.assertRaisesRegex(ValueError, "page-break"):
                    check_author_kit_source(path)
        with self.subTest("captionof"):
            text = "\n".join(
                [
                    r"\usepackage[submission]{AuthorKit27/aaai2027}",
                    r"\captionof{figure}{BGR training loop without domain-specific learner details.}",
                    r"\noindent\textit{Method box: BGR training loop without domain-specific learner details.}",
                ]
            )
            with mock.patch.object(Path, "read_text", return_value=text):
                with self.assertRaisesRegex(ValueError, "page-break"):
                    check_author_kit_source(path)

    def test_author_kit_source_rejects_floating_bgr_algorithm_block(self):
        text = "\n".join(
            [
                r"\usepackage[submission]{AuthorKit27/aaai2027}",
                r"\begin{figure}[t]",
                r"\noindent\textit{Method box: BGR training loop without domain-specific learner details.}",
                r"\end{figure}",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(ValueError, "non-floating"):
                check_author_kit_source(Path("paper/main.tex"))

    def test_author_kit_source_accepts_allowed_packages(self):
        text = "\n".join(
            [
                r"\usepackage[submission]{AuthorKit27/aaai2027}",
                r"\usepackage[hyphens]{url}",
                r"\usepackage{graphicx}",
                r"\usepackage{natbib}",
                r"\noindent\textit{Method box: BGR training loop without domain-specific learner details.}",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=text):
            messages = check_author_kit_source(Path("paper/main.tex"))

        self.assertEqual(messages, ["paper/main.tex: AuthorKit package/page-break/layout rules ok"])

    def test_author_kit_source_requires_method_box_label(self):
        text = r"\usepackage[submission]{AuthorKit27/aaai2027}"
        with mock.patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(ValueError, "missing non-floating BGR method-box label"):
                check_author_kit_source(Path("paper/main.tex"))

    def test_author_kit_source_requires_submission_package(self):
        with mock.patch.object(Path, "read_text", return_value=r"\usepackage{graphicx}"):
            with self.assertRaisesRegex(ValueError, "missing required AAAI AuthorKit"):
                check_author_kit_source(Path("paper/main.tex"))

    def test_author_kit_artifacts_accept_minimal_rebuild_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sty = root / "paper" / "AuthorKit27" / "aaai2027.sty"
            bst = root / "paper" / "AuthorKit27" / "aaai2027.bst"
            sty.parent.mkdir(parents=True, exist_ok=True)
            sty.write_text(
                "\\ProvidesPackage{aaai2027}\n\\bibliographystyle{aaai2027}\n",
                encoding="utf-8",
            )
            bst.write_text("ENTRY {}\nFUNCTION {article} {}\n", encoding="utf-8")
            for relative in ["paper/main.tex", "paper/ReproducibilityChecklist.tex"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(r"\usepackage[submission]{AuthorKit27/aaai2027}", encoding="utf-8")

            self.assertEqual(check_author_kit_artifacts(root), ["AAAI AuthorKit style and BibTeX artifacts ok"])

    def test_author_kit_artifacts_reject_missing_bibtex_style(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sty = root / "paper" / "AuthorKit27" / "aaai2027.sty"
            sty.parent.mkdir(parents=True, exist_ok=True)
            sty.write_text(
                "\\ProvidesPackage{aaai2027}\n\\bibliographystyle{aaai2027}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required AAAI AuthorKit artifact"):
                check_author_kit_artifacts(root)

    def test_author_kit_artifacts_reject_missing_checklist_import(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sty = root / "paper" / "AuthorKit27" / "aaai2027.sty"
            bst = root / "paper" / "AuthorKit27" / "aaai2027.bst"
            sty.parent.mkdir(parents=True, exist_ok=True)
            sty.write_text(
                "\\ProvidesPackage{aaai2027}\n\\bibliographystyle{aaai2027}\n",
                encoding="utf-8",
            )
            bst.write_text("ENTRY {}\nFUNCTION {article} {}\n", encoding="utf-8")
            main = root / "paper" / "main.tex"
            checklist = root / "paper" / "ReproducibilityChecklist.tex"
            main.write_text(r"\usepackage[submission]{AuthorKit27/aaai2027}", encoding="utf-8")
            checklist.write_text(r"\usepackage{graphicx}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required AAAI AuthorKit"):
                check_author_kit_artifacts(root)

    def test_tex_dependencies_required_accepts_current_manifest_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in [
                "paper/main.tex",
                "paper/ReproducibilityChecklist.tex",
                "paper/figures/summary_table.tex",
                "paper/references.bib",
                "paper/AuthorKit27/aaai2027.sty",
                "paper/AuthorKit27/aaai2027.bst",
            ]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            (root / "paper" / "main.tex").write_text(
                "\n".join(
                    [
                        r"\usepackage[submission]{AuthorKit27/aaai2027}",
                        r"\input{figures/summary_table.tex}",
                        r"\bibliography{references}",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "paper" / "ReproducibilityChecklist.tex").write_text(
                r"\usepackage[submission]{AuthorKit27/aaai2027}",
                encoding="utf-8",
            )

            self.assertEqual(
                check_tex_dependencies_required(root),
                ["TeX source dependencies are required submission artifacts ok"],
            )

    def test_tex_dependencies_required_rejects_missing_input_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            main = root / "paper" / "main.tex"
            checklist = root / "paper" / "ReproducibilityChecklist.tex"
            main.parent.mkdir(parents=True, exist_ok=True)
            main.write_text(r"\input{figures/missing_table.tex}", encoding="utf-8")
            checklist.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing TeX dependency artifact"):
                check_tex_dependencies_required(root)

    def test_double_blind_source_rejects_local_paths_and_identity_markers(self):
        email_user = next(
            pattern
            for pattern in DOUBLE_BLIND_FORBIDDEN_PATTERNS
            if pattern.isalpha() and pattern.islower()
        )
        text = f"See {self.author_pattern('/work/')}bgr and contact {email_user}@example.com"
        with mock.patch.object(Path, "read_text", return_value=text):
            with self.assertRaisesRegex(ValueError, "double-blind leak"):
                check_double_blind_source(Path("paper/main.tex"))

    def test_double_blind_source_accepts_anonymous_submission_metadata(self):
        text = "\n".join(
            [
                r"\author{Anonymous Submission}",
                r"\affiliations{Paper under double-blind review}",
                "Boundary-Guided Replay",
            ]
        )
        with mock.patch.object(Path, "read_text", return_value=text):
            messages = check_double_blind_source(Path("paper/main.tex"))

        self.assertEqual(messages, ["paper/main.tex: double-blind source guard ok"])


if __name__ == "__main__":
    unittest.main()
