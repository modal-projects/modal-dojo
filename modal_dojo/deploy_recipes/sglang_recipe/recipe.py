from __future__ import annotations

from dataclasses import dataclass, field

from modal_dojo.common import GPUType
from modal_dojo.deploy_recipes.base import BaseDeployRecipe, DeployRecipeType


@dataclass
class SglangRecipe(BaseDeployRecipe):
    """SGLang server settings.

    Args:
        gpu:
            GPU type for server containers.
        tp:
            Tensor-parallel degree and GPU count. Defaults to one.
        dp:
            Data-parallel degree passed as ``--dp-size``. When set, the recipe also
            passes ``--enable-dp-attention``.
        context_length:
            Maximum context length. The model default applies when unset.
        mem_fraction_static:
            Fraction of GPU memory reserved for model weights and the KV cache.
            SGLang chooses it when unset.
        chunked_prefill_size:
            Chunked-prefill token budget.
        max_running_requests:
            Maximum concurrent requests per worker.
        sglang_image:
            SGLang container image.
        extra_server_args:
            Additional ``sglang.launch_server`` arguments. Use an empty string for
            flag-only arguments.
        env_vars:
            Environment variables merged over the base Hugging Face cache environment.
        install_transformers_from_git:
            Install Transformers from GitHub in the image. Disable this if
            ``sglang_image`` already registers the model architecture because
            duplicate registrations can crash the server.
        environment_name:
            Modal deployment environment.
        deploy_strategy:
            Modal deployment strategy.
        startup_timeout:
            Maximum time in seconds for container startup and the SGLang readiness
            check.
    """

    _recipe_type: DeployRecipeType = field(default=DeployRecipeType.SGLANG, init=False)

    gpu: GPUType = "H100"
    tp: int | None = None
    dp: int | None = None
    context_length: int | None = None
    mem_fraction_static: float | None = None
    chunked_prefill_size: int | None = None
    max_running_requests: int | None = None
    sglang_image: str = "lmsysorg/sglang:v0.5.12"
    extra_server_args: dict[str, str] | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    install_transformers_from_git: bool = True
    environment_name: str | None = None
    deploy_strategy: str = "rolling"
    startup_timeout: int = 20 * 60

    def server_args(self, *, served_model_name: str) -> dict[str, str]:
        """Build SGLang launch arguments.

        Returns:
            Launch arguments keyed by flag name.
        """
        args: dict[str, str] = {"--served-model-name": served_model_name}
        if self.context_length is not None:
            args["--context-length"] = str(self.context_length)
        if self.mem_fraction_static is not None:
            args["--mem-fraction-static"] = str(self.mem_fraction_static)
        if self.chunked_prefill_size is not None:
            args["--chunked-prefill-size"] = str(self.chunked_prefill_size)
        if self.max_running_requests is not None:
            args["--max-running-requests"] = str(self.max_running_requests)
        if self.extra_server_args:
            args.update(self.extra_server_args)
        return args
