---
order: 1
---

# Migrating from Training Gym to Modal Dojo

The Training Gym has been renamed to Modal Dojo. This guide will break down how to migrate your existing Training Gym setup to the new name, as well as some common things to be aware of.

## Package, imports, and environment variables

The repository has been renamed to [modal-dojo](https://github.com/modal-projects/modal-dojo), and the Python package is now imported as `modal_dojo`.

To replace your existing Training Gym dependency:

```bash
uv remove modal-training-gym
uv add git+https://github.com/modal-projects/modal-dojo.git@main
```

Once you have the new package, update your imports to use the new name. In addition to the package itself, be sure to update any symbols that have been renamed as well:

```python
# Before
from modal_training_gym import TrainConfig, TrainingGymError, TrainingGymConfigError

# After
from modal_dojo import TrainConfig, DojoError, DojoConfigError
```

If you're using any `TRAINING_GYM_*` environment variables, be sure to update them to `MODAL_DOJO_*`.

## CLI and dashboard

The `training-gym` command has been renamed to `modal-dojo`, and many other things related to configuration and the dashboard have also been renamed:

* The `~/.training-gym.toml` config file is now `~/.modal-dojo.toml`.
* The `training-gym-overview` skill is now `modal-dojo-overview`.
* The dashboard app, deployed previously as `training-gym-dashboard`, is now `dojo-dashboard`.

To copy your existing configuration over, and to deploy a new `dojo-dashboard` with an updated URL, run:

```bash
modal-dojo setup
```

This command does not stop the old dashboard, so that any ongoing runs can keep reporting their status. Once all active runs are finished, stop the old dashboard:

```bash
modal app stop training-gym-dashboard
```

## Metrics

If you're using Weights & Biases or Trackio, the default project name is now `modal-dojo`.