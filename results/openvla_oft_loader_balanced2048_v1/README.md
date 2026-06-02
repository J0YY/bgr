# OpenVLA-OFT Balanced-2048 Loader Validation

This artifact records the OpenVLA-OFT RLDS loader validation for the matched
BGR-boundary and random-balanced 2,048-step TFDS exports.

Both checks use OpenVLA-OFT's unmodified `make_oxe_dataset_kwargs` and
`make_dataset_from_rlds` on the stock dataset name `libero_goal_no_noops`.

Remote logs:

```text
/work/joy/bgr/logs/run_1780381106_407628579.out
/work/joy/bgr/logs/run_1780381221_375772232.out
```

Both loader checks computed dataset statistics for 2,048 transitions and 32
trajectories, then yielded a 64-step trajectory chunk with primary/wrist image
fields, proprio `(64,8)`, action `(64,7)`, and language.
