import re

from modal_dojo import (
    HuggingFaceDataset,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
    TrainConfig,
)

model = Qwen3_5_4B()


async def gsm8k_rm(args, sample, **kwargs) -> float:
    text = model.parse_response(sample.response or "").content
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    pred = boxed[-1] if boxed else (re.findall(r"-?[\d,]+(?:\.\d+)?", text) or [""])[-1]
    try:
        return float(float(pred.replace(",", "")) == float(sample.label))
    except ValueError:
        return 0.0


config = TrainConfig(
    model=model,
    dataset=HuggingFaceDataset(
        hf_repo="skrishna/gsm8k_only_answer",
        hf_split="train[:120]",
        input_column="text",
        output_column="label",
        input_format="text",
    ),
    recipe=Qwen3_5_4B_Recipe(
        custom_rm_function=gsm8k_rm,
    ),
)

if __name__ == "__main__":
    run = config.launch()
    print(run.training_run_id)
