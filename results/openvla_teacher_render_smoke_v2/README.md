OpenVLA teacher-replay rendering smoke test with one example per perturbation family.

Command:
`scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_render_smoke_v2 --max-examples 4 --selection first_per_family`

This GPU/EGL smoke replays successful native OpenVLA actions in LIBERO, renders the observation stream, applies the candidate perturbation, and writes PNG/action examples for blur, brightness, shift, and occlusion. It validates the next bridge toward RLDS conversion, but it is not a full OpenVLA-OFT fine-tuning run.
