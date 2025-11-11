#!/usr/bin/env python3
"""
Memory-optimized fine-tuning script with LoRA support    # Over    # Override some trainer args for gradient-friendly memory efficiency with 2 GPUs
    train_args.per_device_train_batch_size = 1   # Reduced for gradient memory
    train_args.per_device_eval_batch_size = 1    # Reduced eval batch size
    train_args.gradient_accumulation_steps = 16  # Increased to maintain effective batch size
    train_args.gradient_checkpointing = True
    train_args.bf16 = True                       # Use BF16 for memory efficiency
    train_args.dataloader_num_workers = 0
    train_args.remove_unused_columns = False
    train_args.optim = "adafactor"              # Memory-efficient optimizer for gradients
    train_args.max_grad_norm = 1.0              # Enable gradient clipping for stability
    train_args.eval_steps = 200                 # Enable regular evaluation
    train_args.logging_steps = 25               # Regular logging
    train_args.ddp_find_unused_parameters = False # Optimize DDP performance
    train_args.save_steps = 0                   # Disable checkpoint saving
    train_args.save_strategy = "no"             # Explicitly disable saving
    train_args.save_total_limit = 0             # Don't save any checkpointsiner args for memory efficiency with 2-GPU setup
    train_args.per_device_train_batch_size = 2   # Increase batch size with more memory
    train_args.per_device_eval_batch_size = 2    # Increase eval batch size
    train_args.gradient_accumulation_steps = 8   # Reduce accumulation since batch size is higher
    train_args.gradient_checkpointing = True
    train_args.bf16 = True   # Use BF16 for memory efficiency
    train_args.dataloader_num_workers = 0
    train_args.remove_unused_columns = False
    train_args.optim = "adafactor"  # Memory-efficient optimizer
    train_args.max_grad_norm = 1.0  # Re-enable gradient clipping for stability
    train_args.eval_steps = 500     # Re-enable evaluation
    train_args.logging_steps = 50   # Regular logging wraps the original fine_tune.py with additional memory optimizations.
"""

import sys
import os
import torch
import gc
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🔧 Starting imports...")

# Add the source directory to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))  # Add examples directory to path

print("📦 Importing fine_tune components...")
# Import the original fine-tune components
from fine_tune import parse_args, fine_tune


def setup_memory_optimization():
    """Configure PyTorch for maximum memory efficiency."""
    # Enable memory-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass
    
    # Configure memory management with gradient-specific optimizations
    if torch.cuda.is_available():
        # Clear any existing memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory fraction to leave room for gradients
        torch.cuda.set_per_process_memory_fraction(0.90)  # Reduced from 0.95
        
        # Enable memory mapping for large models with gradient-friendly settings (PyTorch 1.13 compatible)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.8'


def create_lora_model(model, target_modules=None):
    """Apply LoRA to the model for parameter-efficient fine-tuning."""
    if target_modules is None:
        # Default targets for GPT-2 models (adapt based on architecture)
        target_modules = ["c_attn", "c_proj", "c_fc"]
    
    lora_config = LoraConfig(
        r=16,                         # Higher rank since we have more memory
        lora_alpha=32,                # Proportional alpha
        target_modules=target_modules,
        lora_dropout=0.1,             # Slight increase in dropout
        bias="none",                  # No bias adaptation
        task_type=TaskType.CAUSAL_LM  # Task type
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def memory_efficient_fine_tune():
    """Main function with memory optimizations."""
    print("⚙️  Setting up memory optimization...")
    # Setup memory optimization
    setup_memory_optimization()
    
    print("📋 Parsing arguments...")
    # Parse arguments
    args = parse_args()
    model_args, ner_args, train_args, dataset_args, privacy_args, outdir_args, env_args, config_args = args
    
    # Override some trainer args for gradient-friendly memory efficiency with 2 GPUs
    train_args.per_device_train_batch_size = 1   # Reduced for gradient memory
    train_args.per_device_eval_batch_size = 1    # Reduced eval batch size
    train_args.gradient_accumulation_steps = 16  # Increased to maintain effective batch size
    train_args.gradient_checkpointing = True
    train_args.bf16 = True                       # Use BF16 for memory efficiency
    train_args.dataloader_num_workers = 0
    train_args.remove_unused_columns = False
    train_args.optim = "adafactor"              # Memory-efficient optimizer for gradients
    train_args.max_grad_norm = 1.0              # Enable gradient clipping for stability
    train_args.eval_steps = 500                 # Enable regular evaluation
    train_args.logging_steps = 50               # Regular logging
    train_args.ddp_find_unused_parameters = False # Optimize DDP performance
    
    # Add memory-specific training arguments
    if not hasattr(train_args, 'max_steps'):
        train_args.max_steps = 1000
    
    # Re-enable perplexity callbacks with model sharding (less frequent)
    if not hasattr(train_args, 'callback_after_n_steps'):
        train_args.callback_after_n_steps = 1000  # Less frequent than default
    
    # Clear memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("🚀 Starting memory-optimized fine-tuning with LoRA on 2 A100 GPUs...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Available GPUs: {device_count}")
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            print(f"GPU {i} current memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available!")
    
    # Import the components we need - do this after memory setup
    from pii_leakage.models.language_model import LanguageModel
    from pii_leakage.utils.callbacks import EvaluatePerplexityCallback
    
    # With 2 GPUs and model sharding, make perplexity computation memory-efficient
    original_on_step_begin = EvaluatePerplexityCallback.on_step_begin
    
    def memory_efficient_on_step_begin(self, args, state, control, **kwargs):
        # Compute perplexity less frequently and on smaller subset
        if self.num_steps is not None and state.global_step % (self.num_steps * 2) == 0:
            try:
                print(f"🔍 Computing perplexity at step {state.global_step} (reduced frequency and subset)")
                # Limit perplexity computation to smaller subset
                original_dataset = self.dataset
                subset_size = min(100, len(self.dataset["text"]))
                self.dataset = {"text": self.dataset["text"][:subset_size]}
                
                # Call original method
                result = original_on_step_begin(self, args, state, control, **kwargs)
                
                # Restore original dataset
                self.dataset = original_dataset
                return result
            except Exception as e:
                print(f"⚠️  Skipping perplexity computation due to: {e}")
                return control
        return control
    
    EvaluatePerplexityCallback.on_step_begin = memory_efficient_on_step_begin
    
    # Monkey-patch the LanguageModel.load method to add LoRA
    original_load = LanguageModel.load
    
    def lora_load(self, verbose=False):
        # Call original load but with device_map for multi-GPU distribution
        if hasattr(self, '_lm') and self._lm is not None:
            return self  # Already loaded
            
        model_cls, tokenizer = AutoModelForCausalLM, AutoTokenizer
        
        # Multi-GPU device mapping for Llama3-8B
        device_map = {
            "model.embed_tokens": 0,
            "model.norm": 1,
            "lm_head": 1,
        }
        
        # Distribute transformer layers across 2 GPUs
        num_layers = 32  # Llama3-8B has 32 layers
        layers_per_gpu = num_layers // 2
        
        for i in range(num_layers):
            gpu_id = 0 if i < layers_per_gpu else 1
            device_map[f"model.layers.{i}"] = gpu_id
        
        print(f"🔧 Loading model with device_map across 2 GPUs...")
        
        if self.model_args.model_ckpt:
            if verbose:
                print(f"> Loading the provided {self.model_args.architecture} checkpoint from '{self.model_args.model_ckpt}'.")
            
            self._lm = model_cls.from_pretrained(
                self.model_args.model_ckpt, 
                return_dict=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
                low_cpu_mem_usage=True,      # Reduce CPU memory usage during loading
                offload_folder="/tmp/model_offload"  # Temporary offload location
            ).eval()
        elif self.model_args.pre_trained:
            if verbose:
                print(f"> Loading a public, pre-trained {self.model_args.architecture} model.")
            self._lm = model_cls.from_pretrained(
                self.model_args.architecture,
                return_dict=True,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            ).eval()
        else:
            if verbose:
                print(f"> Loading an uninitialized {self.model_args.architecture} model.")
            self._lm = model_cls(config=self.get_config())
        
        # Load tokenizer
        self._tokenizer = tokenizer.from_pretrained(
            self.model_args.architecture,
            use_fast=self.model_args.tokenizer_use_fast
        )
        
        # Apply LoRA to the distributed model
        if self._lm is not None:
            print("📦 Applying LoRA to the distributed model...")
            self._lm = create_lora_model(self._lm)
            
            # Print GPU memory usage after loading
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                cached = torch.cuda.memory_reserved(i) / 1e9
                print(f"GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
        
        return self
    
    # Apply the monkey patch
    LanguageModel.load = lora_load
    
    try:
        fine_tune(model_args, ner_args, train_args, dataset_args, privacy_args, outdir_args, env_args, config_args)
    finally:
        # Restore original methods
        LanguageModel.load = original_load
        EvaluatePerplexityCallback.on_step_begin = original_on_step_begin
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    memory_efficient_fine_tune()