"""Patch Miles rollout to send Gemma-4 VL prompts to SGLang as text.

``miles/rollout/sglang_rollout.py: generate`` runs the HF processor locally and
sends the resulting ``input_ids``. For Gemma-4 those already carry one
``<|image|>`` per vision patch, and SGLang re-validates them against the raw
image, so every request 400s. Turning the chat template off is not an escape:
the processor then gets message dicts and fails in ``validate_inputs``.

slime sends ``text`` instead whenever a single-turn request carries images, so
SGLang expands the placeholders itself; this mirrors that branch. It stays gated
on a Gemma-4 processor because miles expects other models to keep one
placeholder for ``mm_data.py`` to expand later.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

MARKER = "PATCHED_GEMMA4_VL_ROLLOUT_TEXT"

TARGET = pathlib.Path("/root/miles/miles/rollout/sglang_rollout.py")

OLD = """    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids
"""

NEW = f"""    # Use existing tokens for multi-turn or tokenize the new prompt
    if (
        payload.get("image_data")
        and len(sample.response) == 0
        and type(getattr(state, "processor", None)).__name__.startswith("Gemma4")
    ):
        # {MARKER}: Gemma-4's processor pre-expands <|image|> per patch, which
        # SGLang rejects against the single raw image; send text and let it expand.
        payload["text"] = sample.prompt
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids
    elif len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids
"""

if not TARGET.exists():
    print(f"{TARGET} not found; skipping Gemma-4 VL rollout text patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("Gemma-4 VL rollout text patch already applied")
    raise SystemExit(0)

if OLD not in src:
    raise SystemExit(
        "Gemma-4 VL rollout text patch did not match; miles' sglang_rollout.py "
        "payload construction has changed. Re-check it before shipping."
    )

old_at = src.index(OLD)
scope = src[
    max(src.rfind("\ndef ", 0, old_at), src.rfind("\n    def ", 0, old_at)) + 1 : old_at
]

# Matching OLD says nothing about the `state` binding the injected branch reads.
if "state = GenerateState(args)" not in scope:
    raise SystemExit(
        "Gemma-4 VL rollout text patch expects a local `state = GenerateState(args)` "
        "in miles' sglang_rollout.py; it is gone, so the injected branch would never "
        "fire. Re-check how the processor is reached before shipping."
    )

# The gate also reads payload["image_data"], which must already be assigned where
# the replaced block sits.
if 'payload["image_data"]' not in scope:
    raise SystemExit(
        'Gemma-4 VL rollout text patch expects payload["image_data"] to be set '
        "before the token block it replaces in miles' sglang_rollout.py; it is not, "
        "so the injected branch would never fire. Re-check the payload order."
    )

TARGET.write_text(src.replace(OLD, NEW, 1))
print("Patched Gemma-4 VL rollout to send text instead of pre-expanded input_ids")
