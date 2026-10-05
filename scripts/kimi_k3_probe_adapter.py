"""Exercise real adapter updates, validation, installation and decode."""


def check_adapters(client, url, model, tp_size=16):
    import base64
    import hashlib
    import json
    import math
    import pickle
    import time
    from pathlib import Path
    import torch
    from transformers import AutoTokenizer
    from sglang.srt.utils import MultiprocessingSerializer

    cfg = json.loads((Path(model) / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    print("ADAPTER_CONFIG", cfg, flush=True)
    experts = cfg["num_experts"]
    hidden = cfg.get("moe_latent_size", 3584)
    intermediate = cfg["moe_intermediate_size"]
    torch.manual_seed(671)
    # Shared outer factors, expert-specific inner factors, before TP slicing.
    shapes = {
        "gate_proj.lora_A.weight": (1, 32, hidden),
        "gate_proj.lora_B.weight": (experts, intermediate, 32),
        "up_proj.lora_A.weight": (1, 32, hidden),
        "up_proj.lora_B.weight": (experts, intermediate, 32),
        "down_proj.lora_A.weight": (experts, 32, intermediate),
        "down_proj.lora_B.weight": (1, hidden, 32),
    }
    tensors = {
        "base_model.model.language_model.model.layers.1.mlp.experts." + k: torch.randn(
            shape, dtype=torch.bfloat16
        )
        * 0.04
        for k, shape in shapes.items()
    }
    name = "staging_probe"

    def post(path, body, success=True):
        response = client.post(url + path, json=body, timeout=180)
        output = response.json()
        if output is None:
            assert success and path in (
                "/release_memory_occupation",
                "/resume_memory_occupation",
            )
            response.raise_for_status()
            return None
        if success:
            assert response.ok and output.get("success", True), (path, output)
        else:
            assert not output.get("success", True), (path, output)
        return output

    post(
        "/register_lora_adapter",
        {
            "lora_name": name,
            "config_dict": {
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["gate_proj", "up_proj", "down_proj"],
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "bias": "none",
            },
        },
    )
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt = tok.encode("The answer to two plus two is", add_special_tokens=False)

    def generate():
        output = post(
            "/generate",
            {
                "input_ids": prompt,
                "lora_path": name,
                "return_logprob": True,
                "top_logprobs_num": 5,
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
        )
        vals = output["meta_info"]["output_token_logprobs"]
        assert vals and all(math.isfinite(v[0]) for v in vals)
        client.get(url + "/health_generate", timeout=30).raise_for_status()
        return vals

    def send(selected):
        # Plain pickle embeds CPU storage bytes and works across both nodes.
        # This checks adapter transactions, not GPU IPC ownership/copying.
        for key, tensor in selected.items():
            data = base64.b64encode(
                pickle.dumps([(name + ":" + key, tensor)], protocol=5)
            ).decode()
            decoded = MultiprocessingSerializer.deserialize(data)
            assert torch.equal(decoded[0][1], tensor)
            post(
                "/update_weights_from_tensor",
                {"serialized_named_tensors": [data] * tp_size},
            )

    def manifest(selected):
        return {
            name: {
                key: hashlib.sha256(
                    t.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                for key, t in selected.items()
            }
        }

    def begin():
        post("/release_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]})
        post("/begin_weight_update", {"sync_base": False})

    def resume():
        post("/resume_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]})

    initial = generate()
    records = []
    for version in (1, 2):
        tick = time.monotonic()
        update = {k: v * version for k, v in tensors.items()}
        begin()
        send(update)
        post("/end_weight_update", {"expected_lora_checksums": manifest(update)})
        resume()
        observed = generate()
        assert observed != initial, (
            "Installed nonzero adapter had no effect on output logprobs"
        )
        records.append(
            {
                "version": version,
                "seconds": time.monotonic() - tick,
                "logprobs": observed,
            }
        )
        initial = observed
    for case in ("abort", "checksum", "partial"):
        begin()
        update = {k: v * 3 for k, v in tensors.items()}
        if case == "partial":
            update.pop(next(iter(update)))
        send(update)
        body = {"abort": True} if case == "abort" else {}
        if case == "checksum":
            body["expected_lora_checksums"] = manifest(tensors)
        output = post("/end_weight_update", body, success=(case == "abort"))
        resume()
        assert generate() == initial, f"{case} modified the installed adapter"
        records.append({"case": case, "result": output})
    return {
        "passed": True,
        "records": records,
        "unsharded_adapter_bytes": sum(
            t.numel() * t.element_size() for t in tensors.values()
        ),
    }
