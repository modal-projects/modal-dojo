from abc import ABC
from enum import Enum


class DeployRecipeType(Enum):
    VLLM = "vllm"
    SGLANG = "sglang"


class BaseDeployRecipe(ABC):
    _recipe_type: DeployRecipeType
    env_vars: dict[str, str] = {}
