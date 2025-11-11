# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer

from .language_model import LanguageModel


class Llama3(LanguageModel):
    """ A custom convenience wrapper around huggingface Llama-3 utils """

    def get_config(self):
        return LlamaConfig()

    @property
    def tokenizer(self):
        """ Returns this model's tokenizer """
        return self._tokenizer

    @property
    def n_positions(self):
        """ Llama models use max_position_embeddings instead of n_positions """
        return self._lm.config.max_position_embeddings

    def load(self, verbose: bool = False) -> 'LanguageModel':
        """ Loads the model and tokenizer from the checkpoint.
        """
        model_cls, tokenizer = AutoModelForCausalLM, AutoTokenizer

        # Check available GPU memory
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            if verbose:
                print(f"> Available GPU memory: {free_memory_gb:.2f} GB")

        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            raw_ckpt = self.model_args.model_ckpt
            if verbose:
                print(f"> Loading the provided {self.model_args.architecture} checkpoint from '{raw_ckpt}'.")

            # Expand user (~) and check filesystem existence to determine whether
            # this is a local checkpoint or a Hub repo id. Rely on os.path.exists
            # rather than only os.path.isdir because some checkpoints may be
            # files or mounted paths that are not directories from within the job.
            expanded_ckpt = os.path.expanduser(raw_ckpt) if isinstance(raw_ckpt, str) else raw_ckpt

            # If the expanded path exists on this machine, treat it as local.
            # Additionally, treat any absolute path as a local path to avoid
            # Hugging Face Hub validation attempts (which interpret absolute
            # paths as repo ids). We also add a fallback mapping between the
            # cluster GPFS root and the user's home directory if the original
            # path is not present on this node.
            path_exists = isinstance(expanded_ckpt, str) and os.path.exists(expanded_ckpt)
            looks_absolute = isinstance(expanded_ckpt, str) and expanded_ckpt.startswith('/')
            local_only = path_exists or looks_absolute

            # Try common fallback: if path starts with gpfs home and doesn't
            # exist here, try the equivalent /home path, and vice versa.
            if not path_exists and isinstance(expanded_ckpt, str):
                if expanded_ckpt.startswith('/gpfs/home6/'):
                    alt = expanded_ckpt.replace('/gpfs/home6/', '/home/')
                    if os.path.exists(alt):
                        expanded_ckpt = alt
                        path_exists = True
                elif expanded_ckpt.startswith('/home/'):
                    alt = expanded_ckpt.replace('/home/', '/gpfs/home6/')
                    if os.path.exists(alt):
                        expanded_ckpt = alt
                        path_exists = True
                # update local_only if the fallback exists
                local_only = path_exists or looks_absolute

            try:
                self._lm = model_cls.from_pretrained(
                    expanded_ckpt,
                    return_dict=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=local_only,
                ).eval()
            except Exception as e:
                # Provide a clearer error when Transformers treats a local path
                # as a Hub repo id (HFValidationError) or when the path is
                # inaccessible from the compute node.
                raise OSError(
                    f"Failed to load checkpoint from '{raw_ckpt}'.\n"
                    f"Expanded path: '{expanded_ckpt}', local_files_only={local_only}.\n"
                    f"Original error: {e}"
                ) from e

            tokenizer_path = expanded_ckpt
        elif self.model_args.pre_trained:  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(f"> Loading a public, pre-trained {self.model_args.architecture} model.")
            # For Llama3, use the default model identifier
            default_model = "meta-llama/Meta-Llama-3-8B"
            
            # Load with device_map for better memory management
            self._lm = model_cls.from_pretrained(
                default_model, 
                return_dict=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).eval()
            tokenizer_path = default_model
        else:  # no checkpoint and no pre-trained model, hence randomly initialize model's parameters.
            if verbose:
                print(f"> Loading an uninitialized {self.model_args.architecture} model.")
            self._lm = model_cls(config=self.get_config())
            tokenizer_path = "meta-llama/Meta-Llama-3-8B"  # Use default for tokenizer

        # For Llama models, we need to load the tokenizer from the model checkpoint/path
        # If tokenizer file is nested, try common subfolders first (tokenizer, llama3_lora)
        tokenizer_dirs = [tokenizer_path, os.path.join(tokenizer_path, "tokenizer"), os.path.join(tokenizer_path, "llama3_lora"), os.path.join(tokenizer_path, "tokenizer", "tokenizer.json")]
        last_exc = None
        for td in tokenizer_dirs:
            try:
                self._tokenizer = tokenizer.from_pretrained(td, use_fast=self.model_args.tokenizer_use_fast, local_files_only=os.path.isdir(td))
                break
            except Exception as e:
                last_exc = e
                continue
        else:
            # If all attempts failed, raise the last exception with context
            raise RuntimeError(f"Failed to load tokenizer from {tokenizer_path} or common subfolders. Last error: {last_exc}")
        
        # Set pad token for Llama (it doesn't have one by default)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Handle token embedding resizing for Llama models
        num_added_toks = 0
        if '[PAD]' not in self._tokenizer.get_vocab():
            num_added_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if num_added_toks > 0:
            # For Llama models, the embedding layer is at model.embed_tokens
            mean_tok_emb = self._lm.model.embed_tokens.weight.data.mean(dim=0)
            self._lm.resize_token_embeddings(len(self._tokenizer))

            # Initialize the newly-added token embedding to the mean of all token embeddings
            for i in range(num_added_toks):
                self._lm.model.embed_tokens.weight.data[-(i + 1), :] = mean_tok_emb

        # Only move to device if device_map wasn't used (device_map="auto" handles placement automatically)
        if not hasattr(self._lm, 'hf_device_map'):
            try:
                self._lm.to(self.env_args.device)
                if verbose and torch.cuda.is_available():
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_memory_gb = free_memory / (1024**3)
                    print(f"> Model loaded. Available GPU memory: {free_memory_gb:.2f} GB")
            except torch.cuda.OutOfMemoryError as e:
                if verbose:
                    print(f"⚠️  GPU out of memory, falling back to CPU: {e}")
                self._lm.to("cpu")
        
        return self

    def unload(self):
        """ Unload model from memory to free up GPU space """
        if hasattr(self, '_lm') and self._lm is not None:
            del self._lm
            self._lm = None
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
