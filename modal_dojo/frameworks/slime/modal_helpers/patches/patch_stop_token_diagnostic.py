"""Diagnostic patch: trace rollout_stop_token_ids through the entire pipeline.

Instruments three points in slime to log the stop_token_ids value:
1. GenerateState.__init__  — logs args.rollout_stop_token_ids → sampling_params
2. generate()              — logs the HTTP POST payload's stop_token_ids
3. post()                  — logs the raw JSON body sent to SGLang/router

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

# ── Patch 1: GenerateState.__init__ in sglang_rollout.py ─────────────────────
rollout_path = pathlib.Path("/root/slime/slime/rollout/sglang_rollout.py")
if not rollout_path.exists():
    print("WARNING: sglang_rollout.py not found, skipping patch 1")
else:
    src = rollout_path.read_text()
    marker = "PATCHED_STOP_TOKEN_DIAGNOSTIC"
    if marker in src:
        print("sglang_rollout.py already patched for stop-token state diagnostics")
    else:
        # --- Patch GenerateState.__init__ to log stop_token_ids ---
        old_init = "self.sampling_params: dict[str, Any] = dict("
        new_init = (
            f"# {marker}: log stop_token_ids from args\n"
            "        import logging as _stlog\n"
            "        _stlogger = _stlog.getLogger('stop_token_diagnostic')\n"
            "        _stlogger.info(\n"
            "            f'[STOP_TOKEN_DIAG] GenerateState.__init__: '\n"
            '            f\'args.rollout_stop_token_ids={getattr(args, "rollout_stop_token_ids", "ATTR_MISSING")}\'\n'
            "        )\n"
            "        self.sampling_params: dict[str, Any] = dict("
        )
        if old_init not in src:
            print(
                "WARNING: Could not find GenerateState.__init__ sampling_params pattern"
            )
        else:
            src = src.replace(old_init, new_init, 1)

        # --- After the dict construction, log the resulting value ---
        old_after = "spaces_between_special_tokens=False,\n        )"
        new_after = (
            "spaces_between_special_tokens=False,\n        )\n"
            "        _stlogger.info(\n"
            "            f'[STOP_TOKEN_DIAG] GenerateState.sampling_params[\"stop_token_ids\"] = '\n"
            '            f\'{self.sampling_params.get("stop_token_ids", "KEY_MISSING")}\'\n'
            "        )"
        )
        if old_after not in src:
            print("WARNING: Could not find sampling_params dict close pattern")
        else:
            src = src.replace(old_after, new_after, 1)

    # --- Patch generate() to log the HTTP POST payload ---
    payload_marker = "PATCHED_STOP_TOKEN_PAYLOAD_DIAGNOSTIC"
    if payload_marker in src:
        print("sglang_rollout.py already patched for stop-token payload diagnostics")
    else:
        old_payload = '    payload = {\n        "sampling_params": sampling_params,'
        new_payload = (
            f"    # {payload_marker}: log payload stop_token_ids\n"
            "    import logging as _stlog2\n"
            "    _stlog2.getLogger('stop_token_diagnostic').info(\n"
            "        f'[STOP_TOKEN_DIAG] generate() payload: '\n"
            '        f\'sampling_params["stop_token_ids"]={sampling_params.get("stop_token_ids", "KEY_MISSING")}\'\n'
            "    )\n"
            '    payload = {\n        "sampling_params": sampling_params,'
        )
        if old_payload not in src:
            print("WARNING: Could not find payload construction pattern")
        else:
            src = src.replace(old_payload, new_payload, 1)

    rollout_path.write_text(src)
    print("Patched sglang_rollout.py with stop-token diagnostics")

# ── Patch 2: post() in http_utils.py — log the actual JSON body ─────────────
http_path = pathlib.Path("/root/slime/slime/utils/http_utils.py")
if not http_path.exists():
    print("WARNING: http_utils.py not found, skipping patch 2")
else:
    src = http_path.read_text()
    marker2 = "PATCHED_STOP_TOKEN_HTTP_DIAGNOSTIC"
    if marker2 in src:
        print("http_utils.py already patched for stop-token diagnostics")
    else:
        # Patch the _post function to log sampling_params before sending
        old_post = (
            "response = await client.post(url, json=payload or {}, headers=headers)"
        )
        new_post = (
            f"# {marker2}\n"
            "            import logging as _stlog3\n"
            "            if payload and 'sampling_params' in (payload or {}):\n"
            "                _sp = (payload or {}).get('sampling_params', {})\n"
            "                _stlog3.getLogger('stop_token_diagnostic').info(\n"
            "                    f'[STOP_TOKEN_DIAG] HTTP POST to {url}: '\n"
            '                    f\'sampling_params["stop_token_ids"]={_sp.get("stop_token_ids", "KEY_MISSING")}\'\n'
            "                )\n"
            "            response = await client.post(url, json=payload or {}, headers=headers)"
        )
        if old_post not in src:
            print("WARNING: Could not find _post client.post pattern")
        else:
            src = src.replace(old_post, new_post, 1)
        http_path.write_text(src)
        print("Patched http_utils.py with stop-token HTTP diagnostics")

# ── Patch 3: train.py — log parsed args at entry ────────────────────────────
train_path = pathlib.Path("/root/slime/train.py")
if not train_path.exists():
    print("WARNING: train.py not found, skipping patch 3")
else:
    src = train_path.read_text()
    marker3 = "PATCHED_STOP_TOKEN_ARGS_DIAGNOSTIC"
    if marker3 in src:
        print("train.py already patched for stop-token diagnostics")
    else:
        # Find where args = parse_args() is called, and log immediately after
        old_parse = "args = parse_args()"
        new_parse = (
            "args = parse_args()\n"
            f"    # {marker3}\n"
            "    import logging as _stlog4\n"
            "    _stlog4.getLogger('stop_token_diagnostic').info(\n"
            "        f'[STOP_TOKEN_DIAG] train.py: args.rollout_stop_token_ids='\n"
            '        f\'{getattr(args, "rollout_stop_token_ids", "ATTR_MISSING")}\'\n'
            "    )"
        )
        if old_parse not in src:
            print("WARNING: Could not find args = parse_args() pattern in train.py")
        else:
            src = src.replace(old_parse, new_parse, 1)
        train_path.write_text(src)
        print("Patched train.py with stop-token args diagnostics")

print("\n=== Stop token diagnostic patches applied ===")
print("Look for [STOP_TOKEN_DIAG] in logs to trace the value at each stage.")
