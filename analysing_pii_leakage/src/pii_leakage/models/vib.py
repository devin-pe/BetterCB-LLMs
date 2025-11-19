# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from typing import List
from pathlib import Path

from .language_model import LanguageModel, GeneratedText, GeneratedTextList
from ..arguments.sampling_args import SamplingArgs

# Add the disentangling directory to the path to import VIB module
_resolved = None
p = Path(__file__).resolve()
for parent in p.parents:
    if parent.name == 'CB-LLMs':
        disentangling_dir = str(parent / 'disentangling')
        project_root_dir = str(parent)
        _resolved = project_root_dir
        break

# Fallback: try a relative upward path which should point to the repo's disentangling/ folder
if _resolved is None:
    _resolved = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    disentangling_dir = os.path.join(_resolved, 'disentangling')

project_root = _resolved

if disentangling_dir and disentangling_dir not in sys.path:
    sys.path.insert(0, disentangling_dir)
    
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from modules import VIB, VIBConfig
except ImportError as e:
    raise ImportError(f"Failed to import VIB modules from disentangling directory. Error: {e}")


class VIBWrapper(LanguageModel):
    """ A wrapper around VIB (Variational Information Bottleneck) Stage 2 model extending LanguageModel class """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vib = None
        self._stage1_vib = None
        self._use_stage1_cond = True

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
        """ Loads the Stage 2 VIB model and Stage 1 VIB model from checkpoints.
        """
        if not self.model_args.model_ckpt:
            raise ValueError("VIB requires a model checkpoint path (Stage 2)")

        model_ckpt = self.model_args.model_ckpt
        if verbose:
            print(f"> Loading VIB Stage 2 checkpoint from '{model_ckpt}'.")

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
            raise FileNotFoundError(f"VIB Stage 2 checkpoint not found at '{expanded_ckpt}'")

        # Determine base model path (should be fine-tuned Llama3)
        base_model_path = getattr(self.model_args, 'base_model_path', None)
        if base_model_path is None:
            # Default to experiment_00015 if not specified
            base_model_path = os.path.join(project_root, 'analysing_pii_leakage/examples/experiments/experiment_00015')
        
        base_model_path = os.path.expanduser(base_model_path)
        
        # Check base model path exists
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model not found at '{base_model_path}'")

        if verbose:
            print(f"> Loading base Llama3 model from '{base_model_path}'")

        # Load the config and tokenizer
        config = LlamaConfig.from_pretrained(base_model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Set pad token for Llama (it doesn't have one by default)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load the base Llama model
        try:
            self._lm = LlamaModel.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            ).eval()
        except Exception as e:
            raise OSError(f"Failed to load base Llama3 model from '{base_model_path}'. Error: {e}") from e

        device = next(self._lm.parameters()).device if hasattr(self._lm, 'parameters') else self.env_args.device

        # Load Stage 2 VIB model
        if verbose:
            print(f"> Loading Stage 2 VIB model from '{expanded_ckpt}'")

        # Find the model file
        import glob
        model_files = glob.glob(os.path.join(expanded_ckpt, "model*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No VIB model files found in '{expanded_ckpt}'")
        
        stage2_model_path = model_files[0]
        if verbose:
            print(f"> Using Stage 2 model: {stage2_model_path}")

        # Load Stage 2 checkpoint to infer architecture
        stage2_state_dict = torch.load(stage2_model_path, map_location=device)
        
        # Infer cond_dim from checkpoint
        cond_dim = None
        if 'decoder.cond_projection.weight' in stage2_state_dict:
            cond_dim = stage2_state_dict['decoder.cond_projection.weight'].shape[1]
            if verbose:
                print(f"> Inferred cond_dim from checkpoint: {cond_dim}")
        else:
            # No projection layer - determine latent_dim from encoder
            latent_dim = stage2_state_dict['encoder.mu.weight'].shape[0]
            cond_dim = latent_dim
            if verbose:
                print(f"> No projection layer found, cond_dim = latent_dim = {cond_dim}")

        # Infer latent_dim from checkpoint
        latent_dim = stage2_state_dict['encoder.mu.weight'].shape[0]
        
        # Determine layer_weight_averaging
        layer_weight_averaging = 'layer_weights' in stage2_state_dict
        
        if verbose:
            print(f"> Stage 2 latent_dim: {latent_dim}")
            print(f"> Layer weight averaging: {layer_weight_averaging}")

        # Create Stage 2 VIB config
        vib_config = VIBConfig(
            input_dim=config.hidden_size,
            latent_dim=latent_dim,
            stage="2",
            num_classes=self._tokenizer.vocab_size,
            layer_weight_averaging=layer_weight_averaging,
            num_layers=config.num_hidden_layers if layer_weight_averaging else None,
            cond_dim=cond_dim
        )

        self._vib = VIB(vib_config)
        self._vib.load_state_dict(stage2_state_dict)
        self._vib.eval()
        self._vib.to(device)

        if verbose:
            print(f"> Stage 2 VIB model loaded successfully")

        # Load Stage 1 VIB model for conditioning
        stage1_model_path = getattr(self.model_args, 'stage1_model_path', None)
        if stage1_model_path is None:
            # Default path based on Stage 2 path
            stage1_model_path = os.path.join(project_root, 'disentangling/models/vib/1/custom_echr/llama3')
        
        stage1_model_path = os.path.expanduser(stage1_model_path)

        if not os.path.exists(stage1_model_path):
            raise FileNotFoundError(f"Stage 1 model path not found at '{stage1_model_path}'")

        if verbose:
            print(f"> Loading Stage 1 VIB model from '{stage1_model_path}'")

        # Find Stage 1 model file
        stage1_files = glob.glob(os.path.join(stage1_model_path, "model*.pth"))
        if not stage1_files:
            raise FileNotFoundError(f"No Stage 1 model files found in '{stage1_model_path}'")
        
        stage1_file = stage1_files[0]
        if verbose:
            print(f"> Using Stage 1 model: {stage1_file}")

        # Load Stage 1 checkpoint to infer architecture
        stage1_state_dict = torch.load(stage1_file, map_location=device)
        stage1_latent_dim = stage1_state_dict['encoder.mu.weight'].shape[0]
        stage1_layer_averaging = 'layer_weights' in stage1_state_dict

        if verbose:
            print(f"> Stage 1 latent_dim: {stage1_latent_dim}")

        # Create Stage 1 VIB config
        stage1_config = VIBConfig(
            input_dim=config.hidden_size,
            latent_dim=stage1_latent_dim,
            stage="1",
            num_classes=2,  # Binary classification for PERSON detection
            layer_weight_averaging=stage1_layer_averaging,
            num_layers=config.num_hidden_layers if stage1_layer_averaging else None
        )

        self._stage1_vib = VIB(stage1_config)
        self._stage1_vib.load_state_dict(stage1_state_dict)
        self._stage1_vib.eval()
        self._stage1_vib.to(device)

        if verbose:
            print(f"> Stage 1 VIB model loaded successfully")

        return self

    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, sampling_args) -> List[GeneratedText]:
        """ Helper function to generate a single batch of text using VIB Stage 2.
        """
        self._lm.eval()
        self._vib.eval()
        self._stage1_vib.eval()

        generated_texts: List[GeneratedText] = []
        
        # Generate for each input in the batch
        for i in range(input_ids.size(0)):
            single_input = input_ids[i:i+1]  # Keep batch dimension
            
            try:
                # Get Stage 1 conditioning
                # Forward pass through base LM to get hidden states
                with torch.no_grad():
                    outputs = self._lm(single_input, output_hidden_states=True)
                    hidden_states = torch.stack(outputs.hidden_states)  # [num_layers+1, batch, seq_len, hidden_size]
                    # Skip embedding layer and permute
                    hidden_states = hidden_states[1:].permute(1, 0, 2, 3)  # [batch, num_layers, seq_len, hidden_size]
                    
                    # Get Stage 1 conditioning (mu from Stage 1 encoder)
                    stage1_outputs = self._stage1_vib(hidden_states, noise=False)
                    stage1_mu = stage1_outputs[1]  # [batch, seq_len, stage1_latent_dim]
                    
                    # For generation, we typically use the last position's conditioning
                    # But VIB generate expects conditioning for all positions
                    cond = stage1_mu  # [batch, seq_len, stage1_latent_dim]

                # Generate using Stage 2 VIB
                text_ids, vib_outputs = self._vib.generate(
                    single_input,
                    self._lm,
                    cond=cond,
                    mask=attention_mask[i:i+1] if attention_mask is not None else None,
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
                # Fallback to empty generation if VIB generation fails
                print(f"Warning: VIB generation failed: {e}")
                import traceback
                traceback.print_exc()
                generated_texts.append(GeneratedText(text=""))

        return generated_texts

    @torch.no_grad()
    def generate(self, sampling_args: SamplingArgs) -> GeneratedTextList:
        """ Generates text using VIB Stage 2 with Stage 1 conditioning.
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
        """ Print a sample generation """
        self._lm.eval()
        self._vib.eval()
        self._stage1_vib.eval()
        data = self.generate(SamplingArgs(N=1, prompt=prompt, generate_verbose=False, seq_len=64))
        from ..utils.output import print_highlighted
        print_highlighted(data[0].text)
        return data[0].text

    def unload(self):
        """ Unload model from memory to free up GPU space """
        if hasattr(self, '_lm') and self._lm is not None:
            del self._lm
            self._lm = None
        if hasattr(self, '_vib') and self._vib is not None:
            del self._vib
            self._vib = None
        if hasattr(self, '_stage1_vib') and self._stage1_vib is not None:
            del self._stage1_vib
            self._stage1_vib = None
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
