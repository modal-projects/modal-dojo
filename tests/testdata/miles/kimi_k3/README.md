These sources come from `radixark/miles:dev-202609251434`, Miles commit
`41c5e38b94ea23677de93b01a4a77d55677a8f09`.

Refresh inputs with `uv run modal run scripts/fetch_kimi_k3_patch_snapshots.py`.
Outputs apply the corresponding K3 patch:

- `cuda_ipc.py` and `updater.py`: `patch_ipc_bucket_empty_cache`.
- `hf_weight_iterator.py`: `patch_lora_sync_stream_pp`.
- `checkpoint_io.py`: `patch_checkpoint_local_dirs`.

Tests compile patched sources and check idempotence and drift rejection.
The updater lifecycle test verifies that its final bucket is released before
engine commit. Checkpoint tests exercise a writer whose directory is absent
in its local mount, overwrite behavior, completion metadata and write errors.
