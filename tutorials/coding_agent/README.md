# Coding-agent tutorial: full training run

Run these commands from the repository root, with Training Gym configured and
Modal authenticated. This branch configures the full experiment, not a smoke run.

## Dataset sizes used

The full run `convoluted-cove-8024ab40bc4f` used the following SWE-rebench V2 data:

| Stage | Unique tasks | Episodes per task | Dataset filename |
| --- | ---: | ---: | --- |
| Base-model selection probe | 1,000 | 8 | `train-1000.jsonl` |
| Actual training pool | **240** | 8 per sampled prompt | `train-1000-mixed-reward-qwen3-6-27b-agentic-n8.jsonl` |
| Held-out evaluation | **800** | 1 per evaluation | `eval-800.jsonl` |

The selection probe (`excited-sortie-a89b6f9d5494`) generated 8,000 episodes
without updating model weights. A task was retained only if all eight episodes
were gradeable and at least one succeeded and one failed. This probe retained
240 tasks; **the `train-1000` prefix identifies the candidate pool, not the size
of the filtered training dataset**. A fresh probe can select a different count.
The selected IDs and provenance are recorded beside the mixed dataset in its
`.json` file (the dataset itself has a `.jsonl` extension).

The train/eval split groups tasks by repository, with seed 0 and a 20% evaluation
pool. The fixed 800-task evaluation subset is sampled from that separate pool;
it is not selected using the training probe's rewards.

## Prepare or reuse the datasets

Files live under `/data/swe_rebench_v2/` on the `slime-data` Modal Volume. Use the
same Modal environment for preparation and training. For the recorded run:

```bash
export MODAL_ENVIRONMENT=helena-dev
```

If the mixed training dataset and `eval-800.jsonl` already exist in that
environment, reuse them and proceed directly to training. To prepare new data
and run the entire 1,000-task selection probe:

```bash
uv run -m tutorials.coding_agent.dataset
```

This command converts the source tasks, writes train/eval subsets, runs the
8,000-episode probe, and writes the mixed training set. It is substantial GPU and
sandbox work, not a small preprocessing smoke test. It replaces the mixed
selection; skip it when reproducing training with the existing selected tasks.

## Launch the full experiment

```bash
uv run -m tutorials.coding_agent.main
```

Use this module entrypoint directly. It launches the run and prints its training
run ID; the launcher performs model download/conversion if needed.

| Setting | Full-run value |
| --- | --- |
| Initial model | Pretrained `Qwen/Qwen3.6-27B`; no previous training checkpoint specified |
| Maximum training updates | 500 |
| Prompts sampled per update | 32 from the mixed training pool |
| Episodes per sampled prompt | 8 |
| Effective batch | 256 trajectories per update |
| Evaluation | All 800 held-out tasks before training and every 5 updates |
| Evaluation sampling | 1 episode/task, temperature 0.6, top-p 1.0 |
| Checkpoint interval | Every 5 updates |
| Training GPUs | 16 H200: TP 4, PP 2, CP 2 |
| Rollout GPUs | 32 H200: 16 engines with 2 GPUs each |
| Total GPU request | 48 H200 |

The 240 tasks are reused across updates; 256 is the episode batch size, not a
count of distinct training tasks. Dataset preparation also writes small
`train-4`/`eval-4` subsets, but this full configuration does not use them.

The recorded full run stopped on an OOM during update 33; it did not complete
500 updates. This branch patches the pinned Slime loss code during image build
to compute entropy telemetry without retaining backward tensors when
`entropy_coef=0`. Nonzero entropy regularization keeps its gradients. The fix
does not shorten trajectories or alter the effective batch. Full GPU replay of
the failed long-sequence workload is still required to establish memory headroom.
