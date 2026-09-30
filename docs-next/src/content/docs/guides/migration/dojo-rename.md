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

## Dashboard and CLI

The `training-gym` command has been renamed to `modal-dojo`, and the dashboard is now deployed at `dojo-dashboard` instead of `training-gym-dashboard`. Many other things have also been renamed:

* The `~/.training-gym.toml` config file is now `~/.modal-dojo.toml`.
* The `training-gym-overview` skill is now `modal-dojo-overview`.
* Volumes such as `training-gym-metadata` have been renamed to `modal-dojo-metadata`.
* The default Trackio app name is now `modal-dojo-trackio`.

To migrate your configuration and deploy a new `dojo-dashboard` with an updated URL, ensure there are no active training runs, then run:

```bash
modal-dojo migrate
```

The live check reads the training-run summary directly without iterating the volume; evaluations and evaluation results are ignored.

The command will stop old dashboards, move your config file, rename any volumes, and redeploy Modal Dojo apps using their new names.

## Metrics

If you're using Weights & Biases or Trackio, the default project name is now `modal-dojo`.
If the training-run summary is missing or malformed, migration stops. Restore or rebuild the summary before retrying, or manually verify that no training runs are active and run `modal-dojo migrate --force`. The flag skips only the live-run check; all other migration checks remain enabled. Keep the summary current and avoid concurrent launches during migration.
