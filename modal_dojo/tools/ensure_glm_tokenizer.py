"""Ensure GLM-4.7 snapshot has tokenizer.json and tokenizer_config.json.

The HF snapshot for ``zai-org/GLM-4.7`` sometimes lacks ``tokenizer.json``
(only the legacy ``tokenizer.model`` is present).  Slime's
``load_tokenizer`` requires a ``PreTrainedTokenizerFast`` with a chat
template, so we download the missing files into the snapshot directory.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", required=True)
    args = parser.parse_args()

    snapshot_dir: str = args.snapshot_dir
    required = ["tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(snapshot_dir, f))]

    if not missing:
        print("[ensure_glm_tokenizer] All tokenizer files present.")
        return

    from huggingface_hub import hf_hub_download

    for fname in missing:
        dest = hf_hub_download(
            repo_id="zai-org/GLM-4.7",
            filename=fname,
            local_dir=snapshot_dir,
        )
        print(f"[ensure_glm_tokenizer] Downloaded {fname} → {dest}")


if __name__ == "__main__":
    main()
