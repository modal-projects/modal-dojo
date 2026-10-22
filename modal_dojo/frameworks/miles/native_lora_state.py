"""Additional state absent from the pinned native-LoRA checkpoint format."""

from pathlib import Path

import random


def _rank_world():
    import torch.distributed as dist

    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _layout():
    try:
        from megatron.training import get_args
    except ImportError:
        return {}
    args = get_args()
    return {
        key: getattr(args, key, None)
        for key in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "context_parallel_size",
        )
    }


def _rng_state():
    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        state["cuda"] = torch.cuda.get_rng_state_all()
        state["megatron_cuda"] = get_cuda_rng_tracker().get_states()
    return state


def _restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        torch.cuda.set_rng_state_all(state["cuda"])
        get_cuda_rng_tracker().set_states(state["megatron_cuda"])


def capture_resume_state():
    return {"rank_world": _rank_world(), "layout": _layout(), "rng": _rng_state()}


def restore_resume_state(state):
    if state is None:
        raise ValueError("Native LoRA resume requires saved RNG and topology state")
    if tuple(state["rank_world"]) != _rank_world() or state["layout"] != _layout():
        raise ValueError("Native LoRA resume requires the saved rank/topology")
    _restore_rng(state["rng"])


def preserve_resume_flags(args):
    """Remember the requested adapter state before base-weight loading changes flags."""
    if getattr(args, "lora_adapter_path", None):
        args.dojo_native_resume_flags = (args.no_load_optim, args.no_load_rng)


def prepare_native_resume(args):
    flags = getattr(args, "dojo_native_resume_flags", None)
    if flags is None:
        raise ValueError(
            "Native LoRA resume did not preserve the requested optimizer/RNG flags"
        )
    args.no_load_optim, args.no_load_rng = flags


def resume_run_directory(args, rollout_id):
    adapter = getattr(args, "lora_adapter_path", None)
    if not adapter:
        return args.load
    directory = Path(adapter)
    # Native Miles saves the zero-based rollout ID (the older Bridge patch used +1).
    if directory.name != "adapter" or directory.parent.name != f"iter_{rollout_id:07d}":
        raise ValueError(
            "Native adapter checkpoint and restored rollout index disagree"
        )
    return str(directory.parent.parent)
