`upstream.tar.gz` is the unmodified MIT-licensed torch_memory_saver source at
b5588e83de86412a48689a6583a4b567e75f7acc, fetched from the GitHub archive API.
Its license is included in the archive. The test checks its SHA-256 against the
installer pin before patching and compiling the actual allocator core with a
CPU CUDA/disk test double. This verifies lifecycle logic, not GPU protection.

The optional `scripts/check_frozen_weight_backup.py` gate checks real CUDA permissions,
bitwise restoration, adapter changes, logits/logprobs and CUDA graph replay.
Neither test proves full DeepSeek serving compatibility or end-to-end speedup.
