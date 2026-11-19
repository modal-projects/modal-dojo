# ---
# order: 2
# ---
#
# # Training on a Harbor dataset
#
# [Harbor](https://harborframework.com) packages tasks as a directory: an
# `instruction.md`, an `environment/` Dockerfile for the task's container, and a
# `tests/test.sh` that checks the container and writes a reward to
# `/logs/verifier/reward.txt`. That is everything RL needs: a prompt, an
# environment, and a verifier.
#
# `HarborEnvironment` turns a Harbor dataset into an RL environment. The model
# works in the task's own container, a
# [Modal Sandbox](https://modal.com/docs/guide/sandboxes) built from its
# Dockerfile, through an agent harness. When it's done, the task's tests run in
# that same container, and the reward they write is the training reward, the
# same score `harbor run` would give that trajectory.
#
# `HarborEnvironment` uses the [Harbor](https://pypi.org/project/harbor/) and
# [mini-swe-agent](https://pypi.org/project/mini-swe-agent/) packages, so install
# the extra first:
#
# ```bash
# pip install 'modal-dojo[harbor]'
# ```

from modal_dojo import (
    Budget,
    HarborEnvironment,
    ManagedTrainConfig,
    MiniSweAgent,
    Qwen3_5_4B,
)

# ## Choose the tasks and harness
#
# Tasks come from the [Harbor hub](https://hub.harborframework.com/datasets)
# (`name` or `name@version`) or a public GitHub directory, such as
# `https://github.com/laude-institute/harbor/tree/main/examples/tasks/hello-world`.
# Here we use the hub's one-task `hello-world` dataset, which asks for a
# `hello.txt` file.
#
# The `harness` decides how the model works in the container:
#
# - `BashHarness()`: one fenced bash block per reply; it sees the exit code and
#   output.
# - `MiniSweAgent()`: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s
#   text-based agent, with its own prompts: one command per reply, and it
#   submits by printing `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
# - `Terminus2()`: Harbor's Terminus 2 agent: JSON replies send keystrokes to
#   a tmux session.
#
# Each is reproduced turn for turn on the model's own tokens, so the model is
# trained in exactly the harness you'll run it in. To train in a harness of
# your own, subclass `AgentHarness`: its `step` method gets each model reply
# and the episode's sandbox, and returns what the model sees next.
#
# For software-engineering agents, swap in a dataset like `swesmith@1.0` or
# `swe-gen-js@1.0`.

# ## Train
#
# `ManagedTrainConfig` takes a model, an environment, and a `Budget`, and builds
# the rest: it starts from the model's tuned recipe and wires in the
# environment's rollout and reward. Command output is shown to the model but
# masked out of the loss, so only the model's own tokens are trained.
#
# The budget says when to stop: after a number of rollouts, an estimated spend,
# or a number of hours, whichever comes first. Either way the run saves a final
# checkpoint before it stops.
#
# Any recipe field can still be overridden. Here we sample each task 8 times
# per step and turn off Qwen3.5's thinking mode to keep turns short.

config = ManagedTrainConfig(
    Qwen3_5_4B(),
    HarborEnvironment("hello-world", harness=MiniSweAgent(), max_turns=8),
    Budget(max_rollouts=10, max_cost_usd=20),
    recipe_overrides={
        "n_samples_per_prompt": 8,
        "apply_chat_template_kwargs": '{"enable_thinking": false}',
    },
)

if __name__ == "__main__":
    run = config.launch()
    print(f"run id: {run.training_run_id}")
