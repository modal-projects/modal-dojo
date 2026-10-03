"""Validate Kimi-K3 on 64 B300 GPUs; pass --launch to allocate GPUs."""

from configs.kimi_k3_b300 import build_recipe
from scripts.kimi_k3_validation import build_config as _build_config, main


def build_config(**kwargs):
    return _build_config(build_recipe, **kwargs)


if __name__ == "__main__":
    main(build_recipe, gpu_type="B300", context_lengths=(65536, 131072))
