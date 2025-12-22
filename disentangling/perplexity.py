"""
Perplexity evaluation for Stage 2 VIB models with Stage 1 conditioning.
Evaluates language modeling performance on WikiText-103 test set.
"""
import os
import sys
import argparse
import logging
import math
import glob
import shutil
import warnings

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import LlamaConfig, LlamaModel, AutoTokenizer

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
generation_dir = os.path.join(project_root, "generation")
if generation_dir not in sys.path:
    sys.path.insert(0, generation_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import VIB, VIBConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*pad_token_id.*")



def load_wikitext103_texts():
    """Load WikiText-103 test split."""
    texts = []
    
    logger.info("Loading WikiText-103 test split")
    local_candidate = os.path.join("datasets", "wikitext-103", "test", "wiki.test.tokens")
    
    if os.path.exists(local_candidate):
        logger.info(f"Found local WikiText test file at {local_candidate}")
        with open(local_candidate, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or (line.startswith("=") and line.endswith("=")):
                    continue
                texts.append(line)
    else:
        try:
            from datasets import load_dataset
            
            # Clear potentially corrupted cache
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            wikitext_cache = os.path.join(cache_dir, "wikitext")
            if os.path.exists(wikitext_cache):
                logger.info("Clearing WikiText cache")
                shutil.rmtree(wikitext_cache, ignore_errors=True)

            logger.info("Loading wikitext-103 via HuggingFace datasets")
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
            for item in ds:
                line = item["text"].strip()
                if not line or (line.startswith("=") and line.endswith("=")):
                    continue
                texts.append(line)
        except Exception as e:
            logger.error(f"Failed to load wikitext via datasets: {e}")
            raise
    
    return texts



def load_base_model(base_model_path, device):
    """Load the base LlamaModel."""
    logger.info(f"Loading base model from: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    config = LlamaConfig.from_pretrained(base_model_path)
    base_model = LlamaModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    ).eval()
    
    logger.info("Base model loaded successfully")
    return base_model, tokenizer, config


def find_model_file(model_path, postfix):
    """Find model file with given postfix, or return first available .pth file."""
    model_filename = f"model{postfix}.pth"
    full_path = os.path.join(model_path, model_filename)
    
    if os.path.exists(full_path):
        logger.info(f"Found model file: {full_path}")
        return full_path
    
    # Search for alternative models
    logger.warning(f"Model file not found: {full_path}")
    logger.info(f"Searching for alternative model files in {model_path}")
    alt_models = glob.glob(os.path.join(model_path, "model*.pth"))
    
    if alt_models:
        logger.info(f"Found alternative models: {alt_models}")
        selected = alt_models[0]
        logger.info(f"Using: {selected}")
        return selected
    else:
        raise FileNotFoundError(f"No VIB model files found in {model_path}")


def load_stage1_model(stage1_model_path, config, args, device):
    """Load Stage 1 VIB model for conditioning."""
    logger.info(f"Loading Stage 1 VIB model from: {stage1_model_path}")
    
    # Construct Stage 1 model filename
    s1_postfix = f"_bs={args.batch_size}_lr={args.learning_rate}_dim={args.latent_dim}"
    if args.no_ib:
        s1_postfix += "_noib"
    else:
        s1_postfix += f"_b={args.beta_s1}"
    s1_postfix += f"_layer={args.layer_s1}"
    
    stage1_model_file = find_model_file(stage1_model_path, s1_postfix)
    
    # Load checkpoint to infer Stage 1's latent dimension
    stage1_state_dict = torch.load(stage1_model_file, map_location=device)
    stage1_latent_dim = stage1_state_dict['encoder.mu.weight'].shape[0]
    logger.info(f"Inferred Stage 1 latent_dim from checkpoint: {stage1_latent_dim}")
    
    # Create Stage 1 VIB model
    stage1_layer_averaging = (args.layer_s1 == 'all')
    stage1_config = VIBConfig(
        input_dim=config.hidden_size,
        latent_dim=stage1_latent_dim,
        stage="1",
        num_classes=2,  # Binary PII classification
        layer_weight_averaging=stage1_layer_averaging,
        num_layers=config.num_hidden_layers if stage1_layer_averaging else None
    )
    
    stage1_vib = VIB(stage1_config)
    stage1_vib.load_state_dict(stage1_state_dict)
    stage1_vib.eval()
    stage1_vib.to(device)
    
    logger.info("Stage 1 VIB model loaded successfully")
    return stage1_vib, stage1_latent_dim


def load_stage2_model(model_path, config, tokenizer, stage1_latent_dim, args, device):
    """Load Stage 2 VIB model."""
    logger.info(f"Loading Stage 2 VIB model from: {model_path}")
    
    # Construct Stage 2 model filename
    postfix = f"_bs={args.batch_size}_lr={args.learning_rate}_dim={args.latent_dim}"
    if args.no_ib:
        postfix += "_noib"
    else:
        postfix += f"_b={args.beta_s1}_{args.beta_s2}"
    postfix += f"_layer={args.layer_s1}_{args.layer_s2}"
    
    model_file = find_model_file(model_path, postfix)
    
    # Load checkpoint to infer latent dimensions
    vib_state_dict = torch.load(model_file, map_location=device)
    inferred_latent_dim = vib_state_dict['encoder.mu.weight'].shape[0]
    logger.info(f"Inferred Stage 2 latent_dim from checkpoint: {inferred_latent_dim}")
    
    # Infer cond_dim (should match Stage 1 latent_dim)
    if 'decoder.cond_projection.weight' in vib_state_dict:
        cond_dim = vib_state_dict['decoder.cond_projection.weight'].shape[1]
        logger.info(f"Inferred cond_dim from checkpoint: {cond_dim}")
    else:
        # No projection layer means cond_dim equals latent_dim
        cond_dim = stage1_latent_dim
        logger.info(f"No projection layer found, using cond_dim = stage1_latent_dim = {cond_dim}")
    
    # Create Stage 2 VIB model
    layer_weight_averaging = (args.layer_s2 == 'all')
    vib_config = VIBConfig(
        input_dim=config.hidden_size,
        latent_dim=inferred_latent_dim,
        stage="2",
        num_classes=tokenizer.vocab_size,
        layer_weight_averaging=layer_weight_averaging,
        num_layers=config.num_hidden_layers if layer_weight_averaging else None,
        cond_dim=cond_dim
    )
    
    vib_model = VIB(vib_config)
    vib_model.load_state_dict(vib_state_dict)
    vib_model.eval()
    vib_model.to(device)
    
    logger.info("Stage 2 VIB model loaded successfully")
    return vib_model


def compute_perplexity(base_model, vib_model, stage1_vib, tokenizer, texts, device, args):
    """
    Compute perplexity using Stage 2 VIB model with Stage 1 conditioning.
    
    Args:
        base_model: The base LlamaModel
        vib_model: Stage 2 VIB model
        stage1_vib: Stage 1 VIB model for conditioning
        tokenizer: Tokenizer
        texts: List of text strings to evaluate
        device: Device to run computation on
        args: Command line arguments
    
    Returns:
        float: Perplexity value
    """
    base_model.eval()
    vib_model.eval()
    stage1_vib.eval()
    
    # Concatenate all texts
    full_text = " ".join(texts)
    
    # Tokenize
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings['input_ids'].to(device)
    
    max_length = 8192  # Context window size
    seq_len = input_ids.size(1)
    
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    logger.info(f"Computing perplexity for {seq_len} tokens with stride {args.stride}")
    
    # Sliding window evaluation
    for begin_loc in tqdm(range(0, seq_len, args.stride), desc="Computing perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100  # Ignore context tokens
        
        with torch.no_grad():
            # Get features from base model
            outputs = base_model(
                input_chunk,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Transform hidden states
            hidden_states = torch.stack(outputs.hidden_states)
            hidden_states = hidden_states[1:].permute(1, 0, 2, 3)  # (batch, layers, seq, hidden)
            hidden_states = hidden_states.float()  # Convert to float32 for VIB
            
            # Create attention mask
            attention_mask = (input_chunk != tokenizer.pad_token_id).long()
            
            # Get Stage 1 conditioning
            layer_s1 = args.layer_s1 if args.layer_s1 != 'all' else 'all'
            _, cond, _ = stage1_vib(
                hidden_states if layer_s1 == 'all' else hidden_states[:, int(layer_s1):int(layer_s1)+1],
                m=attention_mask,
                noise=False
            )
            
            # Pass through Stage 2 VIB model
            layer_s2 = args.layer_s2 if args.layer_s2 != 'all' else 'all'
            outputs_vib = vib_model(
                hidden_states if layer_s2 == 'all' else hidden_states[:, int(layer_s2):int(layer_s2)+1],
                m=attention_mask,
                cond=cond,
                noise=False
            )
            logits = outputs_vib[0]
            
            # Compute loss for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Only compute loss on valid tokens
            valid_mask = shift_labels != -100
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    shift_logits[valid_mask], 
                    shift_labels[valid_mask], 
                    reduction='sum'
                )
                num_valid_tokens = valid_mask.sum().item()
                
                nll_sum += loss.item()
                n_tokens += num_valid_tokens
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    # Compute final perplexity
    if n_tokens == 0:
        logger.warning("No valid tokens found for perplexity computation")
        return float('inf')
    
    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)
    
    logger.info(f"Average negative log-likelihood: {avg_nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    
    return ppl


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate perplexity for Stage 2 VIB models with Stage 1 conditioning'
    )
    
    # Model paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Stage 2 VIB model checkpoint directory')
    parser.add_argument('--stage1_model_path', type=str, required=True,
                       help='Path to Stage 1 VIB model checkpoint directory')
    parser.add_argument('--base_model_path', type=str, required=True,
                       help='Path to the base LlamaModel (e.g., experiment_00015)')
    
    # Model architecture
    parser.add_argument('--latent_dim', type=int, default=512,
                       help='Latent dimension used during training')
    parser.add_argument('--layer_s1', type=str, default='all',
                       help='Stage 1 layer index or "all" for layer averaging')
    parser.add_argument('--layer_s2', type=str, default='all',
                       help='Stage 2 layer index or "all" for layer averaging')
    
    # Training hyperparameters (for loading correct checkpoint)
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size used during training')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate used during training')
    parser.add_argument('--beta_s1', type=float, default=0.1,
                       help='Beta for Stage 1 used during training')
    parser.add_argument('--beta_s2', type=float, default=0.0,
                       help='Beta for Stage 2 used during training')
    parser.add_argument('--no_ib', action='store_true',
                       help='Whether information bottleneck was disabled during training')
    
    # Evaluation settings
    parser.add_argument('--stride', type=int, default=512,
                       help='Stride for sliding window evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load WikiText-103 dataset
    logger.info("Loading WikiText-103 test dataset")
    texts = load_wikitext103_texts()
    logger.info(f"Loaded {len(texts)} text samples")
    
    # Load base model and tokenizer
    base_model, tokenizer, config = load_base_model(args.base_model_path, device)
    model_device = next(base_model.parameters()).device
    
    # Load Stage 1 VIB model
    stage1_vib, stage1_latent_dim = load_stage1_model(
        args.stage1_model_path, config, args, model_device
    )
    
    # Load Stage 2 VIB model
    vib_model = load_stage2_model(
        args.model_path, config, tokenizer, stage1_latent_dim, args, model_device
    )
    
    # Compute perplexity
    logger.info("Starting perplexity evaluation...")
    perplexity = compute_perplexity(
        base_model, vib_model, stage1_vib, tokenizer, texts, model_device, args
    )
    
    logger.info(f"Final perplexity: {perplexity:.4f}")
    print(f"\nPerplexity: {perplexity:.4f}")
    
    return perplexity


if __name__ == "__main__":
    main()
