import os
import sys
import argparse
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, AutoTokenizer

# Add the generation directory to path for CBL imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import CBL
from config import concepts_from_labels
import config as CFG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_cbllm_perplexity(base_model, cbl_model, tokenizer, texts, device, intervention_vector=None, stride=512):
    """
    Compute perplexity using CB-LLM with concept intervention.
    
    Args:
        base_model: The base LlamaModel
        cbl_model: The CBL module
        tokenizer: The tokenizer
        texts: List of text strings to evaluate
        device: Device to run computation on
        intervention_vector: Optional concept intervention vector
        stride: Stride for sliding window evaluation
    """
    base_model.eval()
    cbl_model.eval()
    
    full_text = " ".join(texts)
    
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings['input_ids'].to(device)
    
    max_length = 8192  # Context window size
    seq_len = input_ids.size(1)
    
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    logger.info(f"Computing perplexity for {seq_len} tokens with stride {stride}")
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            # Get features from base model
            outputs = base_model(input_chunk, use_cache=False)
            features = outputs.last_hidden_state.float()
            
            # Pass through CBL
            concepts, unsup_features, logits = cbl_model(features)
            
            # Apply concept intervention
            if intervention_vector is not None:
                concepts_modified = concepts.clone()
                for j in range(len(intervention_vector)):
                    if j < concepts.size(-1):
                        concepts_modified[:, :, j] = intervention_vector[j]
                e = torch.cat((cbl_model.relu(concepts_modified), unsup_features), dim=-1)
                logits = cbl_model.fc(e)
            
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            # Flatten for loss computation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            valid_mask = shift_labels != -100
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(shift_logits[valid_mask], shift_labels[valid_mask], reduction='sum')
                num_valid_tokens = valid_mask.sum().item()
                
                nll_sum += loss.item()
                n_tokens += num_valid_tokens
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    if n_tokens == 0:
        logger.warning("No valid tokens found for perplexity computation")
        return float('inf')
    
    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)
    
    logger.info(f"Average negative log-likelihood: {avg_nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    
    return ppl


def compute_regular_perplexity(model, tokenizer, texts, device, stride=512):
    """
    Compute perplexity using a regular fine-tuned LlamaForCausalLM model.
    
    Args:
        model: The LlamaForCausalLM model
        tokenizer: The tokenizer
        texts: List of text strings to evaluate
        device: Device to run computation on
        stride: Stride for sliding window evaluation
    """
    model.eval()
    
    full_text = " ".join(texts)
    
    # Tokenize the full text
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings['input_ids'].to(device)
    
    max_length = 8192  # Context window size
    seq_len = input_ids.size(1)
    
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    logger.info(f"Computing perplexity for {seq_len} tokens with stride {stride}")
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            # Get logits directly from the model
            outputs = model(input_chunk, use_cache=False)
            logits = outputs.logits
            
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            # Flatten for loss computation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            valid_mask = shift_labels != -100
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(shift_logits[valid_mask], shift_labels[valid_mask], reduction='sum')
                num_valid_tokens = valid_mask.sum().item()
                
                nll_sum += loss.item()
                n_tokens += num_valid_tokens
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    if n_tokens == 0:
        logger.warning("No valid tokens found for perplexity computation")
        return float('inf')
    
    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)
    
    logger.info(f"Average negative log-likelihood: {avg_nll:.4f}")
    logger.info(f"Perplexity: {ppl:.4f}")
    
    return ppl


def detect_model_type(model_path):
    """
    Detect if the model path contains CBLLM components or is a regular fine-tuned model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        str: 'cbllm' if CBL components found, 'regular' otherwise
    """
    # Check for CBL epoch files
    for epoch in [1, 2]:
        cbl_path = os.path.join(model_path, f"cbl_epoch_{epoch}.pt")
        if os.path.exists(cbl_path):
            return 'cbllm'
    
    # Check for standard model files (pytorch_model.bin or sharded model files)
    model_files = [
        'pytorch_model.bin',
        'pytorch_model-00001-of-*.bin',
        'model.safetensors'
    ]
    
    for pattern in model_files:
        if '*' in pattern:
            # Check for sharded models
            import glob
            matches = glob.glob(os.path.join(model_path, pattern))
            if matches:
                return 'regular'
        else:
            if os.path.exists(os.path.join(model_path, pattern)):
                return 'regular'
    
    # Default to regular if we can't determine
    logger.warning(f"Could not clearly determine model type for {model_path}, assuming regular model")
    return 'regular'


def main():
    parser = argparse.ArgumentParser(description='Evaluate perplexity for CBLLM or regular fine-tuned models')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the model checkpoint (CBLLM or regular fine-tuned)')
    parser.add_argument('--model_type', type=str, choices=['auto', 'cbllm', 'regular'], default='auto',
                       help='Type of model: auto (detect automatically), cbllm, or regular')
    parser.add_argument('--intervention', type=str, default='none',
                       choices=['none', 'one_zero', 'zero_all', 'activate_all'],
                       help='Concept intervention strategy (only for CBLLM)')
    parser.add_argument('--stride', type=int, default=512,
                       help='Stride for sliding window evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from: {args.model_path}")
    
    # Detect model type if auto
    if args.model_type == 'auto':
        detected_type = detect_model_type(args.model_path)
        logger.info(f"Auto-detected model type: {detected_type}")
        args.model_type = detected_type
    
    logger.info(f"Model type: {args.model_type}")
    
    # Load WikiText-103 texts
    logger.info("Loading WikiText-103 dataset")
    texts = load_wikitext103_texts()
    logger.info(f"Loaded {len(texts)} text samples")
    
    # Load model and compute perplexity based on type
    if args.model_type == 'cbllm':
        perplexity = evaluate_cbllm_model(args, device, texts)
    else:
        perplexity = evaluate_regular_model(args, device, texts)
    
    logger.info(f"Final perplexity: {perplexity:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


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
                if not line:
                    continue
                if line.startswith("=") and line.endswith("="):
                    continue
                texts.append(line)
    else:
        try:
            from datasets import load_dataset
            import shutil
            
            # Clear potentially corrupted cache
            cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
            wikitext_cache = os.path.join(cache_dir, "wikitext")
            if os.path.exists(wikitext_cache):
                logger.info("Clearing corrupted WikiText cache")
                shutil.rmtree(wikitext_cache, ignore_errors=True)

            logger.info("Local copy not found; loading wikitext-103 via `datasets`")
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
            for item in ds:
                line = item["text"].strip()
                if not line:
                    continue
                if line.startswith("=") and line.endswith("="):
                    continue
                texts.append(line)
        except Exception as e:
            logger.error(f"Failed to load wikitext via datasets: {e}")
            raise
    
    return texts


def evaluate_cbllm_model(args, device, texts):
    """Evaluate CBLLM model perplexity."""
    try:
        config = LlamaConfig.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Suppress repeated pad_token warnings
        import warnings
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")
        
        base_model = LlamaModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        ).eval()
        
        # Use the correct concept set for CBLLM evaluation
        if 'custom_echr' in concepts_from_labels:
            concept_set = concepts_from_labels['custom_echr']
            logger.info("Using custom_echr concept set from config.py")
        else:
            logger.warning("custom_echr not found in concepts_from_labels, using default concepts")
            concept_set = ['topic', 'sentiment', 'complexity', 'formality', 'technical']  # Fallback
        
        logger.info(f"Using concept set: {concept_set}")
        logger.info(f"Concept dimension: {len(concept_set)}")
        
        
        unsup_dim = CFG.unsup_dim.get('custom_echr', CFG.unsup_dim.get('default', config.hidden_size))
        logger.info(f"Using unsupervised dimension: {unsup_dim}")
        
        cbl_model = CBL(config, len(concept_set), tokenizer, unsup_dim=unsup_dim)

        cbl_path = os.path.join(args.model_path, "cbl_epoch_2.pt") # Change to right file
        
        logger.info(f"Loading CBL weights from: {cbl_path}")
        
        model_device = next(base_model.parameters()).device if hasattr(base_model, 'parameters') else device
        cbl_state_dict = torch.load(cbl_path, map_location=model_device)
        cbl_model.load_state_dict(cbl_state_dict)
        cbl_model.eval()
        
        if not hasattr(base_model, 'hf_device_map'):
            cbl_model.to(device)
        else:
            cbl_model.to(model_device)
        
        logger.info("CBLLM loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load CBLLM: {e}")
        raise
    
    # Set up concept intervention
    intervention_vector = None
    if args.intervention == "one_zero":
        intervention_vector = [1, 0]
        logger.info("Activating concept 0 and deactivating concept 1.")
    elif args.intervention == "zero_all":
        intervention_vector = [0] * len(concept_set)
        logger.info("Using zero_all intervention (suppress all concepts)")
    elif args.intervention == "activate_all":
        intervention_vector = [100] * len(concept_set)
        logger.info("Using activate_all intervention")
    else:
        logger.info("Using no concept intervention")
    
    model_device = next(base_model.parameters()).device if hasattr(base_model, 'parameters') else device
    perplexity = compute_cbllm_perplexity(
        base_model, cbl_model, tokenizer, texts, 
        model_device, intervention_vector, args.stride
    )
    
    return perplexity


def evaluate_regular_model(args, device, texts):
    """Evaluate regular fine-tuned model perplexity."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Suppress repeated pad_token warnings
        import warnings
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")
        
        model = LlamaForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        ).eval()
        
        logger.info("Regular model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load regular model: {e}")
        raise
    
    if args.intervention != 'none':
        logger.warning(f"Intervention '{args.intervention}' specified but model is not CBLLM - ignoring")
    
    model_device = next(model.parameters()).device if hasattr(model, 'parameters') else device
    perplexity = compute_regular_perplexity(
        model, tokenizer, texts, model_device, args.stride
    )
    
    return perplexity


if __name__ == "__main__":
    main()