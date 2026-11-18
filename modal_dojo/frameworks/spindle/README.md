# Spindle connections and model deployments

SpindleRecipe runs a Tinker Cookbook RL configuration from a Modal CPU driver.
The Spindle server owns the trainer and sampler deployments; these persist
across runs and scale according to their Spindle configuration.

```python
from modal_dojo import SpindleRecipe
from tinker_cookbook.rl import train as rl_train

rl_config = rl_train.Config(...)  # unchanged Cookbook configuration

# Connect directly. The API key can also come from TINKER_API_KEY.
recipe = SpindleRecipe(rl_config, base_url=url, api_key=key)

# Or use the saved connection / Dojo-managed server in the active environment.
# Hardware overrides are optional; unspecified settings keep the preset defaults.
recipe = SpindleRecipe(
    rl_config,
    context_length=32768,
    trainer_gpu="H100",
    trainer_gpus_per_node=4,
    inference_gpu="H100",
)
run = recipe.train()
```

A `cookbook_rl(rl_config)` wrapper is also accepted. Custom Tinker programs are
being added separately. This implementation supports synchronous Cookbook RL
and LoRA training.

## Connection selection

1. Explicit URL/key, including the Cookbook URL or TINKER_BASE_URL/TINKER_API_KEY.
2. The `dojo-tinker-default` Modal Secret in the active environment.
3. The compatible managed app, currently `dojo-spindle-v0-1-8`, in that environment.
4. If absent, show the model, context, hardware and environment, then ask before
   deploying a persistent server. In unattended Python, use `.train(yes=True)`
   or `.launch(yes=True)` to approve this step.

Lookup, network and authentication failures stop the launch. They do not trigger
replacement servers. Discovery does not choose an arbitrary Spindle app or scan
other environments. Set the Modal profile and environment before launching.

Managed servers retain their API-key Secret. Future drivers receive that named
Secret and look up the live URL; the credentials need not be downloaded to the
launching computer or written to `~/.modal-dojo.toml`. Explicit connections are
run overrides and do not overwrite the environment's saved default.

To save an external connection deliberately, set TINKER_API_KEY in your shell
(or enter it at the hidden prompt) and run:

```sh
modal-dojo spindle setup --base-url https://your-server.example.com --env dev --yes
```

This creates or updates `dojo-tinker-default` with TINKER_BASE_URL and
TINKER_API_KEY. It is a shared default for authorized users in that environment.
Other Tinker-compatible services use their standard capabilities API; Spindle
provisioning and GPU overrides apply only to Spindle.

## Model deployment selection

Dojo selects a LoRA deployment for the Cookbook model with sufficient total
context capacity. `context_length` includes prompt/history and completion;
Cookbook's `max_tokens` limits generated output. Programs must keep their actual
sequences within the server's capacity.

Without hardware overrides, Dojo reuses the smallest sufficient registered
context. With overrides, it reuses the matching Dojo-created configuration.
Otherwise it selects a bundled Spindle preset, applies the overrides, and adds a
separate deployment through Spindle's deployment API. It never changes an
existing shared backend's hardware. Stable configuration names make later runs
and retries reuse the deployment, including when its apps need redeployment.
The selected deployment name is sent to Tinker model-creation calls while the
original Hugging Face model name remains available for tokenization.

GPU type and count are separate settings. Count, parallelism, concurrency and
other unspecified settings retain preset defaults; arbitrary overrides are not
a guarantee that a model fits in GPU memory. Unsupported model/context choices
fail with instructions to deploy a custom Spindle config.

## Provisioning prerequisites

In this repository, use Python 3.12 and `uv sync --extra spindle-deploy`.
Spindle 0.1.8 is pinned to a commit through `tool.uv.sources`; it requires Modal
SDK 1.6.3. Existing-server clients only need the `spindle` extra unless adding
or selecting a customized backend. Outside a checkout, install the pinned
Spindle source in addition to modal-dojo's spindle extra:

```sh
uv pip install 'modal-dojo[spindle]' 'modal-spindle @ git+https://github.com/modal-projects/spindle.git@16ce201dae9862347cb17437fa0cfed3e3db313e'
```

New servers require Modal proxy credentials permitted in the target environment.
Dojo creates the sampler proxy Secret from those credentials if absent, preserves
existing Secrets, and uses `huggingface-secret` when available. Gated models need
an appropriate HF_TOKEN. Deployment startup and GPU capacity waits are separate
from connection discovery.

You can also prepare the initial server explicitly:

```sh
modal-dojo spindle setup --model Qwen/Qwen3.5-2B --context-length 32768 --env dev --yes
```

Stopping the Dojo run stops its driver, not the reusable Spindle server or its
model deployments. Manage those deployments separately with Spindle/Modal.
