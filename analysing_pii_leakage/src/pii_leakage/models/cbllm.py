# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from typing import List

from .language_model import LanguageModel, GeneratedText, GeneratedTextList
from ..arguments.sampling_args import SamplingArgs

# Add the generation directory to the path to import CBL module.
# Resolve robustly by walking up parents until we find a folder named 'generation'.
from pathlib import Path

_resolved = None
p = Path(__file__).resolve()
for parent in p.parents:
    if parent.name == 'generation':
        # keep both the generation folder and its parent (project root).
        generation_dir = str(parent)
        project_root_dir = str(parent.parent)
        _resolved = project_root_dir
        break
# Fallback: try a relative upward path (4 levels) which should point to the repo's generation/ folder
if _resolved is None:
    _resolved = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

project_root = _resolved
# Insert generation_dir first (so bare `import utils` resolves to generation/utils.py),
# then insert project_root so `import generation` works as a package.
try:
    if 'generation_dir' in locals() and generation_dir not in sys.path:
        sys.path.insert(0, generation_dir)
except Exception:
    pass

if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from generation.modules import CBL
from generation.config import concepts_from_labels
from generation import config as CFG
from generation.utils import top_k_top_p_filtering

class CBLLMWrapper(LanguageModel):
    """ A wrapper around CB-LLM (Concept Bottleneck Language Model) extending LanguageModel class """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cbl = None
        self._concept_set = None
        self._concept_intervention = None

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
        """ Loads the CB-LLM model and tokenizer from the checkpoint.
        """
        if not self.model_args.model_ckpt:
            raise ValueError("CB-LLM requires a model checkpoint path")

        model_ckpt = self.model_args.model_ckpt
        if verbose:
            print(f"> Loading CB-LLM checkpoint from '{model_ckpt}'.")

        # Expand user (~) and check filesystem existence
        expanded_ckpt = os.path.expanduser(model_ckpt) if isinstance(model_ckpt, str) else model_ckpt

        # Check if the path exists and handle common fallbacks
        path_exists = isinstance(expanded_ckpt, str) and os.path.exists(expanded_ckpt)
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

        if not path_exists:
            raise FileNotFoundError(f"CB-LLM checkpoint not found at '{expanded_ckpt}'")

        # Load the config and tokenizer
        config = LlamaConfig.from_pretrained(expanded_ckpt)
        self._tokenizer = AutoTokenizer.from_pretrained(expanded_ckpt)
        
        # Set pad token for Llama (it doesn't have one by default)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load the base Llama model
        try:
            self._lm = LlamaModel.from_pretrained(
                expanded_ckpt,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            ).eval()
        except Exception as e:
            raise OSError(f"Failed to load CB-LLM base model from '{expanded_ckpt}'. Error: {e}") from e

        # Determine concept set based on dataset
        dataset_name = 'custom_echr'  # Default for ECHR dataset
        if dataset_name in concepts_from_labels:
            self._concept_set = concepts_from_labels[dataset_name]
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}' for concept set")

        if verbose:
            print(f"> Using concept set: {self._concept_set}")
            print(f"> Concept dimension: {len(self._concept_set)}")

        unsup_dim = CFG.unsup_dim.get(dataset_name, CFG.unsup_dim.get('default', config.hidden_size))
        if verbose:
            print(f"> Using unsupervised dimension: {unsup_dim}")

        # Load the CBL (Concept Bottleneck Layer)
        self._cbl = CBL(config, len(self._concept_set), self._tokenizer, unsup_dim=unsup_dim)
        
        # Load CBL weights
        cbl_path = os.path.join(expanded_ckpt, "cbl_epoch_2.pt")

        if verbose:
            print(f"> Loading CBL weights from '{cbl_path}'")

        # Load CBL state dict
        device = next(self._lm.parameters()).device if hasattr(self._lm, 'parameters') else self.env_args.device
        cbl_state_dict = torch.load(cbl_path, map_location=device)
        self._cbl.load_state_dict(cbl_state_dict)
        self._cbl.eval()

        # Move CBL to device
        if not hasattr(self._lm, 'hf_device_map'):
            self._cbl.to(self.env_args.device)
        else:
            self._cbl.to(device)

        # Set concept intervention from model args
        self._concept_intervention = getattr(self.model_args, 'concept_intervention', None)
        if verbose and self._concept_intervention:
            print(f"> Concept intervention: {self._concept_intervention}")

        return self

    def _get_intervention_vector(self):
        """ Get the intervention vector based on the concept_intervention setting """
        if not self._concept_intervention or self._concept_intervention == "none":
            return None
        elif self._concept_intervention == "one_zero":
            intervention_vector = [1, 0]
        elif self._concept_intervention == "activate_zero":
            intervention_vector = [100, 0]
        elif self._concept_intervention == "zero_all":
            # Zero out all concepts
            intervention_vector = [0] * len(self._concept_set)
        elif self._concept_intervention == "activate_all":
            # Activate all concepts
            intervention_vector = [100] * len(self._concept_set)
        elif self._concept_intervention.startswith("activate_"):
            # Activate specific concept by index
            try:
                concept_idx = int(self._concept_intervention.split("_")[1])
                if 0 <= concept_idx < len(self._concept_set):
                    intervention_vector[concept_idx] = 100
            except (ValueError, IndexError):
                pass
        return intervention_vector

    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, sampling_args) -> List[GeneratedText]:
        """ Helper function to generate a single batch of text using CB-LLM.
        """
        self._lm.eval()
        self._cbl.eval()

        # Get intervention vector
        intervention_vector = self._get_intervention_vector()

        generated_texts: List[GeneratedText] = []
        
        # Generate for each input in the batch
        for i in range(input_ids.size(0)):
            single_input = input_ids[i:i+1]  # Keep batch dimension
            
            # Use CBL generate method with intervention
            try:
                text_ids, concept_activation = self._cbl.generate(
                    single_input,
                    self._lm,
                    intervene=intervention_vector,
                    length=sampling_args.seq_len,
                    temp=getattr(sampling_args, 'temperature', 0.7),
                    topk=sampling_args.top_k if sampling_args.top_k > 0 else 100,
                    topp=sampling_args.top_p if sampling_args.top_p < 1.0 else 0.9,
                    repetition_penalty=getattr(sampling_args, 'repetition_penalty', 1.5),
                    eos_token_id=self._tokenizer.eos_token_id
                )
                
                # Decode the generated text
                generated_text = self._tokenizer.decode(text_ids[0], skip_special_tokens=False)
                generated_texts.append(GeneratedText(text=generated_text))
                
            except Exception as e:
                # Fallback to empty generation if CBL generation fails
                print(f"Warning: CBL generation failed: {e}")
                generated_texts.append(GeneratedText(text=""))

        return generated_texts

    @torch.no_grad()
    def generate(self, sampling_args: SamplingArgs) -> GeneratedTextList:
        """ Generates text using CB-LLM with concept intervention.
        """
        r = min(self.env_args.eval_batch_size, sampling_args.N)

        # Encode the input prompt
        prompts: List[str] = (
            [" "] * r if sampling_args.prompt is None or sampling_args.prompt.strip() == ""
            else [sampling_args.prompt] * r
        )

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Move inputs to the appropriate device
        if hasattr(self._lm, 'hf_device_map') and self._lm.hf_device_map:
            device = next(self._lm.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        else:
            try:
                device = next(self._lm.parameters()).device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            except StopIteration:
                input_ids = input_ids.to(self.env_args.device)
                attention_mask = attention_mask.to(self.env_args.device)

        generated_data: List[GeneratedText] = []
        num_batches = int((sampling_args.N + self.env_args.eval_batch_size - 1) // self.env_args.eval_batch_size)
        
        for _ in range(num_batches):
            generated_data.extend(self.generate_batch(input_ids, attention_mask, sampling_args))

        return GeneratedTextList(data=generated_data[:sampling_args.N])

    def print_sample(self, prompt=None):
        """ Print a sample generation with concept intervention """
        self._lm.eval()
        self._cbl.eval()
        data = self.generate(SamplingArgs(N=1, prompt=prompt, generate_verbose=False, seq_len=64))
        from ..utils.output import print_highlighted
        print_highlighted(data[0].text)
        return data[0].text

    def unload(self):
        """ Unload model from memory to free up GPU space """
        if hasattr(self, '_lm') and self._lm is not None:
            del self._lm
            self._lm = None
        if hasattr(self, '_cbl') and self._cbl is not None:
            del self._cbl
            self._cbl = None
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()