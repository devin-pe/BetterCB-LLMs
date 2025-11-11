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

# Add the generation directory and current directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
generation_dir = os.path.join(project_root, "generation")

if generation_dir not in sys.path:
    sys.path.insert(0, generation_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules import VIB, VIBConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_vib_perplexity(base_model, vib_model, stage1_vib, tokenizer, texts, device, layer_idx='all', stride=512, use_stage1_cond=False):
    """
    Compute perplexity using VIB model with optional Stage 1 conditioning.
    
    Args:
        base_model: The base LlamaModel
        vib_model: The VIB module (Stage 2)
        stage1_vib: Stage 1 VIB model for conditioning (can be None)
        tokenizer: The tokenizer
        texts: List of text strings to evaluate
        device: Device to run computation on
        layer_idx: Layer index or 'all' for layer averaging
        stride: Stride for sliding window evaluation
        use_stage1_cond: Whether to use Stage 1 conditioning
    """
    base_model.eval()
    vib_model.eval()
    if stage1_vib is not None:
        stage1_vib.eval()
    
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
            outputs = base_model(
                input_chunk,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Transform hidden states
            hidden_states = torch.stack(outputs.hidden_states)
            hidden_states = hidden_states[1:].permute(1, 0, 2, 3)  # (batch, layers, seq, hidden)
            
            # Create attention mask
            attention_mask = (input_chunk != tokenizer.pad_token_id).long()
            
            # Get Stage 1 conditioning if needed
            cond = None
            if use_stage1_cond and stage1_vib is not None:
                _, cond, _ = stage1_vib(
                    hidden_states if layer_idx == 'all' else hidden_states[:, int(layer_idx):int(layer_idx)+1],
                    m=attention_mask,
                    noise=False
                )
            
            # Pass through VIB model
            outputs_vib = vib_model(
                hidden_states if layer_idx == 'all' else hidden_states[:, int(layer_idx):int(layer_idx)+1],
                m=attention_mask,
                cond=cond,
                output_attentions=False,
                noise=False
            )
            logits = outputs_vib[0]
            
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
    Detect if the model path contains VIB components or is a regular fine-tuned model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        str: 'vib' if VIB model files found, 'regular' otherwise
    """
    # Check for VIB model files (from training_cbllm.py)
    import glob
    
    # Look for model*.pth files (VIB format)
    vib_patterns = [
        'model_bs=*_lr=*_dim=*.pth',
        'model*.pth'
    ]
    
    for pattern in vib_patterns:
        matches = glob.glob(os.path.join(model_path, pattern))
        if matches:
            logger.info(f"Found VIB model file: {matches[0]}")
            return 'vib'
    
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
    parser = argparse.ArgumentParser(description='Evaluate perplexity for VIB or regular fine-tuned models')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the VIB model checkpoint directory')
    parser.add_argument('--base_model_path', type=str, required=True,
                       help='Path to the base LlamaModel (e.g., experiment_00015)')
    parser.add_argument('--model_type', type=str, choices=['auto', 'vib', 'regular'], default='auto',
                       help='Type of model: auto (detect automatically), vib, or regular')
    parser.add_argument('--stage', type=str, choices=['1', '2'], default='2',
                       help='VIB stage (1 or 2)')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='Latent dimension for VIB')
    parser.add_argument('--layer', type=str, default='all',
                       help='Layer index or "all" for layer averaging')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size used during training (for loading correct checkpoint)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate used during training (for loading correct checkpoint)')
    parser.add_argument('--beta_s1', type=float, default=0.1,
                       help='Beta for stage 1 (for loading correct checkpoint)')
    parser.add_argument('--beta_s2', type=float, default=0.1,
                       help='Beta for stage 2 (for loading correct checkpoint)')
    parser.add_argument('--no_ib', action='store_true',
                       help='Whether information bottleneck was disabled during training')
    parser.add_argument('--use_stage1_cond', action='store_true',
                       help='Use Stage 1 conditioning for Stage 2 evaluation')
    parser.add_argument('--stage1_model_path', type=str, default=None,
                       help='Path to Stage 1 VIB model (required if use_stage1_cond=True)')
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
    logger.info(f"Loading VIB model from: {args.model_path}")
    logger.info(f"Loading base model from: {args.base_model_path}")
    
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
    if args.model_type == 'vib':
        perplexity = evaluate_vib_model(args, device, texts)
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


def evaluate_vib_model(args, device, texts):
    """Evaluate VIB model perplexity."""
    try:
        # Load tokenizer and base model
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Suppress warnings
        import warnings
        warnings.filterwarnings("ignore", message=".*pad_token_id.*")
        
        config = LlamaConfig.from_pretrained(args.base_model_path)
        base_model = LlamaModel.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        ).eval()
        
        logger.info("Base model loaded successfully")
        
        # Determine number of classes based on stage
        if args.stage == '1':
            num_classes = 2  # Binary PII classification
        else:
            num_classes = tokenizer.vocab_size  # Language modeling
        
        # Create VIB config
        layer_weight_averaging = (args.layer == 'all')
        vib_config = VIBConfig(
            input_dim=config.hidden_size,
            latent_dim=args.latent_dim,
            stage=args.stage,
            num_classes=num_classes,
            layer_weight_averaging=layer_weight_averaging,
            num_layers=config.num_hidden_layers if layer_weight_averaging else None
        )
        
        vib_model = VIB(vib_config)
        
        # Construct model filename based on training parameters
        postfix = f"_bs={args.batch_size}_lr={args.learning_rate}_dim={args.latent_dim}"
        if args.no_ib:
            postfix += "_noib"
        else:
            if args.stage == '1':
                postfix += f"_b={args.beta_s1}"
            else:
                postfix += f"_b={args.beta_s1}_{args.beta_s2}"
        postfix += f"_layer={args.layer}"
        
        model_filename = f"model{postfix}.pth"
        model_path = os.path.join(args.model_path, model_filename)
        
        logger.info(f"Loading VIB model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            logger.info(f"Looking for alternative model files in {args.model_path}")
            import glob
            alt_models = glob.glob(os.path.join(args.model_path, "model*.pth"))
            if alt_models:
                logger.info(f"Found alternative models: {alt_models}")
                model_path = alt_models[0]
                logger.info(f"Using: {model_path}")
            else:
                raise FileNotFoundError(f"No VIB model files found in {args.model_path}")
        
        model_device = next(base_model.parameters()).device if hasattr(base_model, 'parameters') else device
        vib_state_dict = torch.load(model_path, map_location=model_device)
        vib_model.load_state_dict(vib_state_dict)
        vib_model.eval()
        vib_model.to(model_device)
        
        logger.info("VIB model loaded successfully")
        
        # Load Stage 1 model if needed
        stage1_vib = None
        if args.use_stage1_cond and args.stage == '2':
            if args.stage1_model_path is None:
                logger.error("--use_stage1_cond requires --stage1_model_path")
                raise ValueError("Stage 1 model path required for conditioning")
            
            logger.info(f"Loading Stage 1 VIB model from: {args.stage1_model_path}")
            
            stage1_config = VIBConfig(
                input_dim=config.hidden_size,
                latent_dim=args.latent_dim,
                stage="1",
                num_classes=2,
                layer_weight_averaging=layer_weight_averaging,
                num_layers=config.num_hidden_layers if layer_weight_averaging else None
            )
            stage1_vib = VIB(stage1_config)
            
            # Construct Stage 1 model filename
            s1_postfix = f"_bs={args.batch_size}_lr={args.learning_rate}_dim={args.latent_dim}"
            if args.no_ib:
                s1_postfix += "_noib"
            else:
                s1_postfix += f"_b={args.beta_s1}"
            s1_postfix += f"_layer={args.layer}"
            
            stage1_model_path = os.path.join(args.stage1_model_path, f"model{s1_postfix}.pth")
            
            if not os.path.exists(stage1_model_path):
                logger.error(f"Stage 1 model not found: {stage1_model_path}")
                import glob
                alt_s1_models = glob.glob(os.path.join(args.stage1_model_path, "model*.pth"))
                if alt_s1_models:
                    logger.info(f"Found alternative Stage 1 models: {alt_s1_models}")
                    stage1_model_path = alt_s1_models[0]
                    logger.info(f"Using: {stage1_model_path}")
                else:
                    raise FileNotFoundError(f"No Stage 1 model found in {args.stage1_model_path}")
            
            stage1_state_dict = torch.load(stage1_model_path, map_location=model_device)
            stage1_vib.load_state_dict(stage1_state_dict)
            stage1_vib.eval()
            stage1_vib.to(model_device)
            logger.info("Stage 1 VIB model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load VIB model: {e}")
        raise
    
    model_device = next(base_model.parameters()).device if hasattr(base_model, 'parameters') else device
    perplexity = compute_vib_perplexity(
        base_model, vib_model, stage1_vib, tokenizer, texts,
        model_device, args.layer, args.stride, args.use_stage1_cond
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