---
name: autoconfig
description: Turn a training intent ("a 27B Qwen with 32k context on SWE-bench") into a planned, validated batch of Modal Dojo sweeps with `modal-dojo autoconfig`. Use when asked to pick a model and recipe flags, compare configurations, or kick off sweeps without hand-writing TrainConfigs.
---

# Autoconfig

`modal-dojo autoconfig` is the agent's interface to the dashboard's Autoconfig
page. You, the agent, are the "model that decides the flags": the CLI gives you
the registry of models, each recipe's knobs with defaults and docs, a free
dry-run with a GPU-memory estimate per variant, and a launcher that only
accepts a plan that fits. Every run it launches is a normal Modal Dojo
training run, so the `agent-driven-training` skill applies once runs exist.

Only the `sweep` and `status` commands talk to the dashboard (`modal-dojo
setup` must have run). `models` (without history), `knobs`, and `plan` work
offline and never spend GPU time.

## The loop

1. **Parse the intent** into: model family and size, context length, dataset,
   objective (fastest step, cheapest step, best reward), and budget if given.
2. **Pick the model**: `modal-dojo autoconfig models --json`. Match on `name`
   (e.g. "qwen 27b" → `Qwen3.8-27B` and `Qwen3.6-27B`). When several match,
   prefer the newest generation unless the user named one, and say which you
   chose. Catalogue entries (`in_catalogue: true`) have tuned recipes and
   history (`history.cost_per_step_usd`, `time_per_step_s`, `initial_reward`);
   other registry models use their framework's base recipe.
3. **Read the knobs**: `modal-dojo autoconfig knobs <model> --json`. Use the
   `highlight`ed fields first; `--all` lists everything. Flag paths are
   `TrainConfig` dotted paths, almost always `recipe.<field>`.
4. **Plan**: `modal-dojo autoconfig plan -m <model> --context 32k --dataset swe-bench --set … --grid … --json`.
   Read every variant's `memory` (`peak_gib` vs `gpu_gib`, `fits`, `drivers`)
   and `context_plan`. Fix the shape until `fits` is true for every variant
   you want to launch; a bad field path fails with the valid fields listed.
5. **Launch**: the same flags with `sweep` instead of `plan`, plus `--name`
   and `--json`. It plans again, refuses OOM variants (no `--allow-oom` unless
   the user accepts the risk), posts to `/api/autoconfig/sweeps`, and prints an
   `operation_id`. `--wait` polls until every run has launched.
6. **Monitor**: `modal-dojo autoconfig status <operation_id> --json` gives the
   `training_run_id` of each launched run and per-variant launch failures.
   From there use `modal-dojo run get/logs/trace` per the
   `agent-driven-training` skill. Runs share a group id (`autoconfig-<name>-…`)
   and show on the Autoconfig and Training runs pages.

Report what you launched as a plan the user can audit: model, recipe, dataset,
steps, the fixed flags, the grid, GPUs per run, estimated memory, and the
operation/group ids. Launching is not success; a run has to train.

## Choosing flags

- **Context length** (`--context 32k`) sets `recipe.rollout_max_response_len`
  (and `eval_max_response_len`) and raises `recipe.max_tokens_per_gpu` to at
  least that many tokens so one sample packs on one GPU. Memory grows roughly
  linearly with `max_tokens_per_gpu`; when the estimate no longer fits, in
  order: a larger GPU (`recipe.gpu_type`: H100 < H200 < B200 < B300), context
  parallelism (`recipe.context_parallel_size=2,4`, which divides the per-GPU
  token budget), tensor parallelism (`recipe.tensor_model_parallel_size`), more
  nodes (`recipe.actor_num_nodes`), `recipe.recompute_granularity=full`, or
  `recipe.optimizer_cpu_offload=true`. Check the plan after each change.
- **Topology** fields must stay consistent: `tensor_model_parallel_size ×
  context_parallel_size × pipeline` divides the actor GPU count; `plan` reports
  the recipe's own validation errors when they do not.
- **Throughput and cost**: `recipe.rollout_batch_size`,
  `recipe.n_samples_per_prompt`, and `recipe.global_batch_size` trade step time
  against samples per step; `recipe.rollout_num_gpus_per_engine` sizes the
  inference engine. Cheaper GPUs lower `$/step` only while the model fits.
- **Optimisation** sweeps worth running first: `recipe.lr` (e.g.
  `5e-7,1e-6,2e-6`), `recipe.eps_clip`, `recipe.kl_loss_coef`,
  `recipe.rollout_temperature`.
- **Steps**: keep `--steps` at the default 3 for shape/cost comparisons; raise
  it only once a shape is proven, and only to what the user asked for.
- **Datasets**: `--dataset swe-bench` (SWE-smith problem statements → patches)
  and `gsm8k` are presets; any Hugging Face repo works with `--input-column`
  and `--output-column`. There is no SWE-bench grader in the package yet, so on
  SWE-bench the reward column reflects the recipe's default reward.

## Sweep design

- One concern per sweep: a shape sweep (GPU type, parallelism, packing) on the
  default steps first, then an optimisation sweep on the winning shape.
- Keep grids small; the run count is `models × product(grid)`. `plan` prints
  it. Five to eight runs is a batch; more needs the user's agreement.
- Put values you want on every run in `--set`, values you want compared in
  `--grid`. `--grid` wins when both name the same path.
- Values parse as JSON (`1e-6`, `true`, `"B300"` or `B300`), so quote strings
  with spaces.

## Example

Intent: "a qwen 27b with 32k context on swe bench, try a few learning rates".

```bash
modal-dojo autoconfig models --json | jq '.[] | select(.name | test("27B"))'
modal-dojo autoconfig knobs Qwen3.8-27B --json | jq '.knobs[] | select(.highlight)'
modal-dojo autoconfig plan -m Qwen3.8-27B --context 32k --dataset swe-bench \
  --grid recipe.lr=5e-7,1e-6,2e-6 --json
# every variant fits on 1x B300 → launch
modal-dojo autoconfig sweep -m Qwen3.8-27B --context 32k --dataset swe-bench \
  --grid recipe.lr=5e-7,1e-6,2e-6 --name qwen27b-32k-lr --wait --json
modal-dojo run get <training_run_id>
```

## Guardrails

- Never pass `--allow-oom` on your own initiative.
- Do not raise `--steps`, add models, or widen the grid beyond the intent to
  "be safe"; ask.
- If `plan` fails on a path, read the valid fields it lists rather than
  guessing another spelling.
- When the dashboard is unreachable, say so and stop at the plan; do not fall
  back to launching `TrainConfig`s directly.
