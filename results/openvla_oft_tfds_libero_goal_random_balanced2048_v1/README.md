# OpenVLA-OFT LIBERO-Named Random-Balanced-2048 TFDS

This remote artifact exports the matched random-balanced 2,048-step rendered
set under the stock OpenVLA-OFT dataset name `libero_goal_no_noops`, so the
unmodified OpenVLA-OFT LIBERO registry can load it.

Generation command:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 01:00:00 /work/joy/bgr /work/joy/safesae-openvla/bin/python scripts/export_openvla_oft_tfds.py --examples runs/openvla_teacher_oft_random_balanced2048_v1/examples.jsonl --out runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1 --dataset-name libero_goal_no_noops --version 1.0.0
```

OpenVLA-OFT loader smoke:

```bash
~/remote_srun.sh --github-test --git-pull --log --partition compute --gres '' --cpus 4 --mem 16G --time 00:30:00 /work/joy/bgr env PYTHONPATH=/work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft /work/joy/external_validation/openvla_oft_smoke_746850/openvla-oft/.venv-oft/bin/python -c "from pathlib import Path; import tensorflow_datasets as tfds; from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs; from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds; root=Path('/work/joy/bgr/runs/openvla_oft_tfds_libero_goal_random_balanced2048_v1'); kw=make_oxe_dataset_kwargs('libero_goal_no_noops', root, load_camera_views=('primary','wrist'), load_proprio=True, load_language=True); ds, stats=make_dataset_from_rlds(train=True, shuffle=False, **kw); ex=next(iter(tfds.as_numpy(ds))); print('keys', sorted(ex.keys())); print('obs', {k:v.shape for k,v in ex['observation'].items()}); print('task', ex['task']['language_instruction'][0].decode()); print('action', ex['action'].shape); print('stats_action_mean', stats['action']['mean'].shape); print('stats_proprio_mean', stats['proprio']['mean'].shape); print('num_transitions', stats.get('num_transitions')); print('num_trajectories', stats.get('num_trajectories'))"
```

Remote logs:

```text
/work/joy/bgr/logs/run_1780380831_695183835.out
/work/joy/bgr/logs/run_1780381221_375772232.out
```

The loader smoke computes OpenVLA-OFT dataset statistics for 2,048 transitions
and 32 trajectories. The yielded trajectory chunk has primary/wrist image
fields, proprio `(64,8)`, action `(64,7)`, and language.
