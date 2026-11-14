# ---
# order: 12
# ---
#
# # Training an agent harness on Harbor tasks

from pathlib import Path

from modal_dojo import (
    HarborAgent,
    HarborTasks,
    Qwen3_5_4B,
    Qwen3_5_4B_Miles_Recipe,
    TrainConfig,
)

from tutorials.pi_harbor.tasks import write_tasks

TASKS = Path(__file__).parent / "tasks"
CONTEXT_LENGTH = 16384
MAX_REPLY_TOKENS = 1024

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=HarborTasks(agent=HarborAgent("pi"), path=str(TASKS), task_glob="train-*"),
    eval_dataset=HarborTasks(agent=HarborAgent("pi"), path=str(TASKS), task_glob="eval-*"),
    recipe=Qwen3_5_4B_Miles_Recipe(
        docker_image="radixark/miles:dev-202610140350@sha256:c53f65d346a6b364cf89d7dbaa416595a2d5638670b5d82c683214183dca10bd",
        miles_git_ref="d33406f1139f1fa66305b17b9e91dfa110e3c4ce",
        ref_load="/checkpoints/Qwen3.5-4B_torch_dist_miles_d33406f",
        environment={
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "CONVERT_KEEP_PP1": "1",
        },
        gpu_type="H200",
        actor_num_gpus_per_node=8,
        num_rollout=50,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=128,
        rollout_max_response_len=MAX_REPLY_TOKENS,
        max_tokens_per_gpu=CONTEXT_LENGTH,
        apply_chat_template_kwargs='{"enable_thinking": false}',
        sglang_tool_call_parser="qwen3_coder",
        eval_interval=4,
        n_samples_per_eval_prompt=1,
        eval_max_response_len=MAX_REPLY_TOKENS,
        save_interval=8,
        extra_config={
            "max_seq_len": CONTEXT_LENGTH,
            "tito_model": "qwen35",
        },
    ),
)

if __name__ == "__main__":
    write_tasks(TASKS)
    run = config.launch()
    print(f"run id: {run.training_run_id}")
