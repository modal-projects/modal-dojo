from __future__ import annotations

import datetime
import hashlib


def create_hash(
    model_name: str,
    checkpoint: str,
    recipe: str,
    app_name: str,
    model_path: str,
) -> str:
    import randomname

    created_at = str(int(datetime.datetime.now(datetime.UTC).timestamp()))

    slug = randomname.get_name(sep="-").lower()

    hash_parts = [
        model_name,
        checkpoint,
        recipe,
        app_name,
        model_path,
        created_at,
    ]
    suffix_payload = "\x1f".join(hash_parts)
    suffix = hashlib.sha256(suffix_payload.encode()).hexdigest()[:12]
    return f"{slug}-{suffix}"
