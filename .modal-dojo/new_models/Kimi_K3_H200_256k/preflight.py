"""Offline capacity-probe checks with the real tokenizer and mocked generation.

Run with --tokenizer PATH --dataset-arrow PATH. This does not validate GPU
execution or the upstream mathematical reward implementation.
"""

import argparse
import asyncio
import json
import sys
import tempfile
import types
from pathlib import Path

import validate
from datasets import Dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset-arrow", required=True)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = list(Dataset.from_file(args.dataset_arrow).select(range(4)))
    prompt_data = Path(tempfile.mkdtemp()) / "prompts.jsonl"
    prompt_data.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))
    names = (
        "miles",
        "miles.rollout",
        "miles.rollout.base_types",
        "miles.rollout.generate_utils",
        "miles.rollout.generate_utils.generate_endpoint_utils",
        "miles.rollout.rm_hub",
        "miles.utils",
        "miles.utils.http_utils",
        "miles.utils.lora",
        "miles.utils.lora.utils",
        "miles.utils.types",
    )
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["miles.rollout.base_types"].GenerateFnOutput = types.SimpleNamespace
    sys.modules[
        "miles.rollout.generate_utils.generate_endpoint_utils"
    ].compute_routing_headers = lambda *_: {}
    sys.modules["miles.utils.lora.utils"].LORA_ADAPTER_NAME = "preflight"

    class Sample:
        class Status:
            COMPLETED = "completed"
            TRUNCATED = "truncated"

        def update_from_meta_info(self, args, info):
            self.status = self.Status.COMPLETED

    sys.modules["miles.utils.types"].Sample = Sample
    reports = []
    # Exercise interrupted thinking and each possible completed boundary.
    endings = (
        "",
        "<|close|>think",
        "<|close|>response",
        "<|close|>message",
        "<|end_of_msg|>",
    )
    for i, ending in enumerate(endings):
        row = rows[i % len(rows)]
        prompt = tok.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            thinking_effort="low",
        )
        sample = Sample()
        sample.index, sample.prompt, sample.label = i, prompt, row["label"]
        sample.metadata = {}
        requests = []

        async def post(url, body, **kwargs):
            ids = list(body["input_ids"])
            budget = body["sampling_params"]["max_new_tokens"]
            assert len(ids) + budget < 262144
            if not requests:
                response = "I will solve the problem." + ending
            else:
                assert len(ids) == 261120
                response = "The answer is \\boxed{42}."
                rendered = tok.decode(ids, skip_special_tokens=False)
                assert "Original problem, repeated for reference:" in rendered
                assert "End diagnostic observation." in rendered
                assert row["prompt"][-1]["content"] in rendered
            requests.append({"input_tokens": len(ids), "budget": budget})
            tokens = tok.encode(response, add_special_tokens=False)
            return {
                "text": response,
                "meta_info": {
                    "output_token_logprobs": [[-0.1, t] for t in tokens],
                    "finish_reason": {"type": "stop"},
                },
            }

        sys.modules["miles.utils.http_utils"].post = post
        input = types.SimpleNamespace(
            args=types.SimpleNamespace(
                sglang_context_length=262144,
                k3_validation_mode="capacity",
                sglang_router_ip="localhost",
                sglang_router_port=1,
                prompt_data=str(prompt_data),
                input_key="prompt",
            ),
            sample=sample,
            state=types.SimpleNamespace(tokenizer=tok),
            sampling_params={},
        )
        asyncio.run(validate.generate_probe(input))
        assert len(requests) == 2
        assert 261120 <= len(sample.tokens) < 262144
        assert set(sample.loss_mask) == {0, 1}
        assert all(
            p == 0
            for p, mask in zip(sample.rollout_log_probs, sample.loss_mask)
            if not mask
        )
        assert sample.metadata["k3_generated_tokens"] == sum(sample.loss_mask)
        # Test the reward wrapper's data isolation, without pretending to test
        # the mocked upstream math scorer's correctness.
        calls = []
        sys.modules["miles.rollout.rm_hub"].get_deepscaler_rule_based_reward = (
            lambda response, label: calls.append((response, label)) or 0.0
        )
        for response in ("\\boxed{42}", "\\boxed{-1}", "\\boxed{", ""):
            sample.metadata["k3_final_response"] = response
            asyncio.run(validate.score_final_response(None, sample))
            assert calls[-1] == (response, sample.label)
        sample.metadata.pop("k3_final_response")
        asyncio.run(validate.score_final_response(None, sample))
        assert calls[-1] == ("", sample.label)
        reports.append(
            {
                "ending": ending,
                "total_tokens": len(sample.tokens),
                "generated_tokens": sum(sample.loss_mask),
                "requests": requests,
            }
        )

    invalid = Sample()
    invalid.tokens, invalid.response_length = [], 0
    invalid.rollout_log_probs, invalid.loss_mask = [], []
    for tokens, probs in (([1], []), ([1], [float("nan")]), ([1], [float("inf")])):
        try:
            validate.append_tokens(invalid, tokens, probs, generated=True)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid token/logprob data accepted")
    config = validate.build_config(validate.build_recipe)
    assert config.recipe.gpu_allocation.total_gpus == 128
    for concurrency in (1, 2, 4, 8):
        config = validate.build_config(
            validate.build_recipe, concurrency_per_engine=concurrency
        )
        assert config.recipe.sglang_server_concurrency == concurrency
        assert config.recipe.sglang_max_running_requests == concurrency
        assert config.recipe.sglang_max_mamba_cache_size == 5 * concurrency
    report = {
        "status": "passed",
        "real_dataset_rows": len(rows),
        "scope": "Offline real-tokenizer, mocked-generation contract check; no GPU or math-reward validation",
        "cases": reports,
    }
    Path(__file__).with_name("preflight.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
