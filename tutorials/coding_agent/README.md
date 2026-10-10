# Coding-agent tutorial: full training run

Run these commands from the repository root, with Modal Dojo configured and
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

## Evaluate without training

The evaluation patch limits the full eval to 128 simultaneous episodes (the
previous configuration allowed 512). Task IDs, temperature 0.6, top-p 1.0,
8,192 output tokens per turn, 75 turns, and the 1,800-second agent budget stay
the same. The 600-second generation request timeout is unchanged.

On a recoverable generation transport failure, evaluation runs the verifier
against the surviving sandbox before cleanup. It records the failure phase,
exception, request duration, input token count, and request token budget.
Recovered grades are marked separately. A verifier failure remains ungraded;
it never receives credit merely because generation recovery was attempted.
Training and the dataset selection probe do not enable grade recovery.

New eval metrics include `graded_count`, `ungraded_count`,
`generation_error_count`, `recovered_grade_count`, and `solve_rate_on_graded`.
The last metric is diagnostic; retain the full-set score and disclose invalid
episodes rather than silently dropping hard tasks from the denominator.

Start with the four-task proof (24 H200 GPUs, zero optimizer updates):

```bash
uv run -m tutorials.coding_agent.evaluate --proof
```

Then compare both arms on the same 800 held-out tasks. Each command launches
an evaluation-only job with the original 48-H200 topology and no checkpoint
writes or optimizer updates. Wait for each job to finish before starting the
next one:

```bash
uv run -m tutorials.coding_agent.evaluate
uv run -m tutorials.coding_agent.evaluate \
  --checkpoint /checkpoints/convoluted-cove-8024ab40bc4f \
  --checkpoint-step 29
```

Saved iteration 29 is the checkpoint after 30 updates. Add `--dry-run` to
inspect either configuration without launching, or `--concurrency N` for a
controlled serving-load comparison. Infrastructure retries are disabled for
these diagnostic runs; inspect failures before launching another job.

### Recorded evaluation validation (October 9–10, 2026)

The four-task proof `worn-tag-b51a5149b56c` completed with four valid grades,
two solves, and no generation errors. The full comparison used the commands
above at concurrency 128, with the same 800 task IDs in both saved artifacts:

| Weights | Run ID | Solved / 800 | Valid grades | Generation errors |
| --- | --- | --- | --- | --- |
| Pretrained | `pounded-electricity-02a362e0622f` | 302 (37.75%) | 798 | 0 |
| After 30 updates, saved iteration 29 | `rectilinear-school-f04a611b283c` | 315 (39.375%) | 799 | 0 |

Startup logs confirmed the pretrained checkpoint at iteration 0 and the
training checkpoint at iteration 29. Both jobs performed zero optimizer
updates and their Modal apps stopped after completion.

The net gain was 13 tasks: 46 new passes and 33 regressions. This single
sampled comparison does not establish a reliable improvement (paired exact
McNemar p=0.177; approximate task-level 95% interval for the difference:
−0.55 to +3.80 percentage points). These uncertainty calculations do not model
correlation between tasks from the same repository.

Ungraded episodes were verifier timeouts: `getmoto__moto-7607` in both arms,
and `getmoto__moto-7608` in the baseline. They remain zero in the full-set
score. Both tasks passed with their supplied reference fixes in separate
sandbox checks. Restricting to the 798 tasks graded in both arms still gives
302 versus 315 solves.

Mean generated output grew from 10,661 to 18,939 tokens per episode. Episodes
ending on an 8,192-token generation with no usable tool call grew from 5 to 29. The
framework labels these `ContextLengthExceeded`, but the saved tails show a
per-turn output limit, not proof that the full context window was exhausted.
The reliability fix therefore does not explain away the weak held-out gain;
length usage and transfer across task types remain learning diagnostics.

Grade recovery was separately verified by injecting a generation timeout
after a correct Click patch: the real sandbox verifier still returned a valid
passing grade and retained the generation-error metadata. Neither full run
needed recovery. The patch's 11 local tests also passed.
