# Muse Spark Open: H200 and B300 64K validation report

Status snapshot: 2026-10-16, approximately 19:11 UTC. Branch: `helena/muse-spark-open`.

The H200 and B300 recipes are implemented, registered, and locally tested. Native
GPU probes pass on both architectures after repairing the upstream image. Neither
hardware variant has completed a full-model rollout or optimizer step at 64K.
These are candidate recipes, not production-validated configurations.

## Current iteration

The current H200 proof is `stale-trumpet-bc6d2db1f110`, app
[`ap-N0KtuyOJrNu3iXCeBhnD6C`](https://modal.com/apps/modal-labs/training-team/ap-N0KtuyOJrNu3iXCeBhnD6C).
It was launched at 19:08 UTC from commit `16e6103`, with a 32-H200 conversion
and the unchanged 64-H200 training topology. No model step has completed.

The preceding proof, `quiet-thinner-c6ecd7dbfbcb` / `ap-jRiZ6TCnh6SNyDzCewGUp7`,
waited from 18:45 to 19:07 UTC without a single GPU worker. It was deliberately
stopped; its conversion call was confirmed terminated before releasing its lock.
The fresh conversion uses TP1/PP4/EP8, matching upstream's B300 conversion layout.
A CPU inspection of all pinned HF tensor headers estimates at most 56.626 GiB of
text weights per rank, excluding runtime buffers and frozen omni companions.
This supports trying four H200 nodes without needing all eight training nodes
for preparation; it does not prove runtime fit or guarantee faster scheduling.
The conversion has 1 TiB of node-local disk. See `conversion_footprint.json`.

The user authorized fixes and validation through successful H200 and B300 smokes,
including Autoconfig registration. Stopped the stale B300 controller and its
dependent H200 waiter before starting this attempt; all old queued conversion
calls were confirmed terminated before releasing their checkpoint lock. The
local launch uses Autoconfig's planner/tags and the current Dojo sources, avoiding
relaunches caused by preemption of the earlier remote controller. B300 will reuse
only a complete checkpoint after H200 preparation.

Commit `94856dc` fixes generated documentation and removes redundant recipe
declarations without changing effective settings. The full local suite passes
(1,415 tests, one skipped, four subtests), all six Autoconfig integration tests
pass, and GitHub Tests passed on `94856dc`. The four-node conversion change
passes 100 relevant Dojo tests and all six Autoconfig integration tests; the new
GitHub Tests and formatting checks also pass on `16e6103`. The separate Miles patch snapshot workflow
fails on a pre-existing mismatch: the default image at commit
`ab48452eaf35ef6beaf9c745e1c923bf261bd235` does not contain the expected
`miles/ray/rollout/rollout_manager.py`, and several other fixture hashes also differ.
The default image and snapshot script are unchanged by this branch. GPU CI model
validation still requires a targeted run after the live recipe proofs succeed.

Earlier submissions and dependency-controller logic are preserved in
`h200_validation_launch.json`, `launch_h200_validation.py`, and `live_status.json`.
Those files are historical evidence, not the currently active launcher.

## Files and registration

- `modal_dojo/common/models/muse_spark.py`: pinned model config.
- `modal_dojo/train_recipes/miles_recipe/muse_spark.py`: both
  `Muse_Spark_Open_64k_Recipe` (H200) and
  `Muse_Spark_Open_64k_B300_Recipe` (B300).
- `modal_dojo/train_recipes/miles_recipe/muse_spark_image.py`: native image repairs
  and HF cache compatibility patch.
- Both recipes are exported from `modal_dojo`. H200 is the default recipe for this
  model and is registered as `Muse-Spark-Open` in the model validation registry.
- Companion autoconfig branch `helena/muse-spark-open` contains the `dapo-math`
  workload, seeded 65,536-token config group, and `examples/muse_spark_64k.py`.

[Autoconfig dashboard](https://modal-labs-training-team--autoconfig-fastapi-app.modal.run/group/meta-models-muse-spark-open-816b-a42b--dapo-math--65536).

## Run snapshot before the new H200 submission

Actual Modal call outcomes take precedence over persisted run status: some startup
failures leave the metadata row marked `running`.

| Run | Hardware | Observed state | App |
| --- | --- | --- | --- |
| `asymptotic-similarity-89353a872d87` | B300 | Newer conversion queued; cached model download finished; no containers assigned at inspection. Owns the shared-volume conversion lock. | [ap-Di9KJVzXKFcqo3mrAcm6Eh](https://modal.com/apps/modal-labs/training-team/ap-Di9KJVzXKFcqo3mrAcm6Eh) |
| `espressivo-geyser-54387bcb1c34` | H200 | Follow-up failed during checkpoint resolution because the B300 run already owns the conversion lock. No training was submitted. | [ap-9zdAFstFJ2NcK8RKa2ftSZ](https://modal.com/apps/modal-labs/training-team/ap-9zdAFstFJ2NcK8RKa2ftSZ) |
| `foggy-flask-1432d863d70e` | H200 | Training failed before Ray startup on a missing offline HF `main` reference. Conversion returned success, but subsequent integrity verification found missing shards. | [ap-H4anHKCcZikyPZe8egnBXB](https://modal.com/apps/modal-labs/training-team/ap-H4anHKCcZikyPZe8egnBXB) |
| `celeste-perimeter-b08ea29fcbe6` | B300 | Prior conversion call is terminated; persisted run status is stopped. The launch graph also contains the newer pending conversion listed above. | [ap-98eIK0yn8Thg7T5OedmCWo](https://modal.com/apps/modal-labs/training-team/ap-98eIK0yn8Thg7T5OedmCWo) |

The older run table is historical: the queued B300 conversion and its controllers
have since been stopped. Only the current attempt described above is active.

## Recipe settings

| Setting | H200 | B300 |
| --- | --- | --- |
| GPUs | 64, eight nodes of eight | 32, four nodes of eight |
| Training TP / PP / CP | 4 / 4 / 4 | 8 / 4 / 1 |
| EP / expert TP | 16 / 1 | 8 / 1 |
| Conversion TP / PP / EP | 1 / 4 / 8 (32 GPUs) | 1 / 4 / 8 (32 GPUs) |
| Rollout GPUs per engine | 16 | 8 |
| Training attention probe | FlashAttention 3 | FlashAttention 2; TE FA4 disabled |
| Node-local training disk | 2 TiB | 3 TiB |
| Node-local conversion disk | 1 TiB | 1 TiB |

Both recipes use a 65,536-token context: 4,096 prompt tokens plus up to 61,440
response tokens. Sequence length, maximum positions, and SGLang context are all
65,536. They use BF16 weights/KV cache, full recomputation, streamed Adam state on
local disk, GRPO with four prompts and four samples per prompt, global batch 16,
and one rollout/update by default. Weight and optimizer checkpoint saving are
disabled for the proof runs. Ten-step smokes are authorized after the one-step
proofs succeed; no smoke is claimed as completed. Longer training is out of scope.

Both candidate recipes use `name="miles-muse_spark_open_64k_b300_recipe"`, selecting
`miles-muse_spark_open_64k_b300_recipe-checkpoints`, and
`ref_load="/checkpoints/Muse-Spark-Open-816B-A42B_torch_dist"`.
Only one conversion may write that shared checkpoint at a time. Training may
read it concurrently after completeness verification. Already submitted Modal
functions retain their original serialized configs and code.

## Upstream provenance and pins

- Model: `meta-models/Muse-Spark-Open-816B-A42B`.
- HF revision: `0fb02478b02565b2ce7fc57e6051c91172774ac9`.
- Miles: `753e19d8bd8fe5a27892dd18b787dc131addd269`.
- SGLang: `ad34947129c518641db51816c88c126ca16f786c`.
- Megatron: `6530834aaaaf3f25e3cd08dfe4678834c6b1a28e`.
- Pinned base image: `radixark/miles:muse-spark-open@sha256:2b46ca326003e347c600145bd60935b42af8032ed6f0a75e7fdc7bf5a8700b15`, with the native repair overlay.

[Miles PR 4111](https://github.com/radixark/miles/pull/4111) reports upstream
30-step validation on 64 H200s and 32 B300s at an 8K context. That does not establish
64K correctness or capacity. [Documentation PR 4112](https://github.com/radixark/miles/pull/4112)
names the image tag. [Image commit 563e297](https://github.com/radixark/miles/commit/563e297d7f3206eea8edcac04cbb877471e4343b)
pins the validated B300 sources and disables `SGLANG_ENABLE_BF16_SPLITK_GEMM`
because its experimental CUTLASS dependency is unavailable. Both recipes now
carry that setting.

[Build 38833380524](https://github.com/radixark/miles/actions/runs/38833380524)
published the newer tag with amd64 digest
`sha256:8fc0c6efc99f5da12823f579e244f9eae841a55f58cd54c021db293aaa03f977`.
No exact validation-run image digest or public W&B run link was located. A CPU
audit of that newer image still found FlashAttention 2 and Apex native imports
failing with an undefined c10 CUDA symbol. The recipes therefore retain the
already tested repaired image, rather than treating the moving tag as verified.

## Failures found and fixes

1. **Native image ABI mismatch.** The vendor image combined Torch 2.14.1+cu130
   with native wheels built for Torch 2.13. The overlay rebuilds Transformer
   Engine's Torch extension, FlashAttention 2, and Apex against the installed
   Torch, while preserving the upstream TE core changes and FA3. B300 TE FA4
   backward also failed with the installed Quack combination; trainer attention
   uses the tested FA2 path, while SGLang's FA4 remains available.
2. **HF cache shard layout.** Upstream's downloader uses a local model directory.
   Dojo uses a shared HF cache, including snapshot-to-repository-blob-to-global-
   blob symlinks. The upstream loader rejected valid shards outside the snapshot
   directory. The patch accepts only the canonical cache layout, validates paths,
   and uses the validated targets for reads. Traversal and unrelated symlink
   targets remain rejected.
3. **Conversion release publication.** Upstream moved its release rename into
   `publish_as_release`. The wrapper now suppresses both the old inline rename
   and the newer helper for multi-node conversion, retaining `iter_0000001` and
   tracker `1` so a head-local rename cannot erase other nodes' shard paths.
4. **Pinned revision at training startup.** Conversion/download used the pinned
   revision, but training prewarm and a bridge fallback requested `main`. Both
   lookups now propagate the pinned model revision. Regression tests reproduce
   an offline cache containing only the pinned snapshot and no `refs/main`.
5. **Incomplete checkpoint accepted as complete.** H200's converted metadata
   references 64 shard files, but only 56 were committed. Ranks 40–47 are missing,
   corresponding to one node. Present shard bytes: 1,429,560,273,018; referenced
   required bytes: 1,640,366,059,810. Do not reuse that H200 volume. Conversion
   now sets `MODAL_DOJO_CHECKPOINTS_VOLUME_NAME`, activating the existing
   all-rank commit synchronization; the head reloads before checking. Cache-hit,
   partial-cleanup, and post-conversion checks now validate every metadata storage
   range against the referenced file's size. This fix is locally tested but has
   not yet completed a fresh multi-node conversion.

## Validation evidence and limits

- Real cached HF weights: all 65 shard headers and the 2,000-entry tensor index
  were inspected; a small real tensor read was finite. Total cached bytes:
  1,644,333,393,864. Probe app: `ap-3iOrxNo7oLx1rfw1P7oonE`.
- Final repaired Dojo image: one H200 and one B300 passed native imports,
  global/sliding-window attention forward/backward against FP32 SDPA,
  TE GroupedLinear backward, Apex optimizer update, and 64K argument parsing.
  Probe app: `ap-JG12XVG13IDsMWKqaCvpIr`.
- Model/tokenizer/provider and representative DAPO formatting preflight passed:
  `ap-j7Xm8cS30UdGyAg0QO4hVj`. Example prompt lengths: 323, 270, and 287 tokens.
- The newer upstream image ABI audit ran as `ap-zc24rGScLvyl6YrMCgRmk4`.
- Local tests cover recipe registration, CLI flags, HF cache symlinks, release
  publication, pinned offline revision resolution, missing/truncated shards,
  commit error propagation, and conversion verification ordering. Exact latest
  local test totals are recorded in `checks.json`.
- No full-model reward, generated response quality, optimizer-step timing,
  throughput, or 64K memory fit has been established. Native microprobes and
  successful argument parsing are not a full-model proof.

## Next validation steps

Monitor the current H200 conversion to a terminal state and verify every
referenced shard before reuse. Then complete the one-step H200 and B300 proofs
against that shared checkpoint. Require sensible responses, nonzero raw reward,
an optimizer update, and step/substep timing. Follow each successful proof with
a fresh ten-step smoke, inspect early and late traces and rewards, and verify the
successful runs and metrics in Autoconfig. Check actual sequence lengths before
claiming empirical long-context fit; configuring a 65,536-token cap alone is not
that evidence.

The evidence files contain status and probe results only. Authentication tokens,
secret values, full model weights, and unrelated workspace changes are excluded.
