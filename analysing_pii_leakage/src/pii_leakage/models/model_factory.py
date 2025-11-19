# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from .gpt2 import GPT2
from .llama3 import Llama3
from .vib import VIBWrapper
from .language_model import LanguageModel


class ModelFactory:
    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> LanguageModel:
        if "opt" in model_args.architecture:
            raise NotImplementedError
        elif "gpt" in model_args.architecture:
            return GPT2(model_args=model_args, env_args=env_args)
        elif "llama" in model_args.architecture:
            return Llama3(model_args=model_args, env_args=env_args)
        elif "cbllm" in model_args.architecture:
            # Lazy import to avoid loading generation dependencies unless actually using CBLLM
            from .cbllm import CBLLMWrapper
            return CBLLMWrapper(model_args=model_args, env_args=env_args)
        elif "vib" in model_args.architecture:
            return VIBWrapper(model_args=model_args, env_args=env_args)
        else:
            raise ValueError(f"Unsupported architecture: {model_args.architecture}")
