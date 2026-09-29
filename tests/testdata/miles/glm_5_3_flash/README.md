# GLM-5.3-Flash patch snapshots

The `train.py`, `actor.py`, and `model.py` inputs were read on Modal from Miles
commit `cc76e23915b2132ecfff65b97331fd83b02637e6` (the merge of PR #2786).
Their source paths are `/root/miles/train.py` and
`/root/miles/miles/backends/megatron_utils/{actor,model}.py`.

`chunk_intra_token_parallel.py.input` is from FLA 0.4.2 at
`/usr/local/lib/python3.12/dist-packages/fla/ops/kda/chunk_intra_token_parallel.py`
inside `radixark/miles:glm53next`, OCI index digest
`sha256:66725f740a6013b00d27e21fdfd480a24b0d3b5c61840342e9405bf1e09e5162`.

The corresponding `.output` files are the recipe-specific adapters' expected
output. When updating the image or Miles pin, fetch these four sources from
the candidate image, apply `patch_glm_5_3_flash_{timing,kda}.patch_source`, and
review the resulting golden changes. Run
`uv run pytest tests/test_glm_5_3_flash_patches.py` to check the contract.
