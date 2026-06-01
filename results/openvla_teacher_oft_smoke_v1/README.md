OpenVLA teacher-replay rendering smoke with OpenVLA-OFT LIBERO fields.

Command:
`scripts/render_openvla_teacher_examples.py --manifest results/openvla_teacher_replay_manifest_v1/teacher_replay_manifest.jsonl --out runs/openvla_teacher_oft_smoke_v1 --max-examples 4 --selection first_per_family`

This GPU/EGL smoke replays successful native OpenVLA actions in LIBERO, renders primary and wrist images, records the 8D LIBERO state used by OpenVLA-OFT (`EEF_state` plus gripper state), and saves action/language labels in NPZ records. It covers blur, brightness, shift, and occlusion. It validates the field-level bridge toward RLDS conversion, but it is not a full OpenVLA-OFT fine-tuning run.
