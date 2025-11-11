import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import LlamaConfig, LlamaModel, AutoTokenizer

# Add paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
generation_dir = os.path.join(project_root, "generation")

if generation_dir not in sys.path:
    sys.path.insert(0, generation_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import VIB, VIBConfig
from utils import top_k_top_p_filtering
import time

parser = argparse.ArgumentParser(description='Generate text using Stage 2 VIB with Stage 1 conditioning')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
parser.add_argument("--base_model_path", type=str, required=True,
                   help="Path to base LlamaModel (e.g., experiment_00015)")
parser.add_argument("--stage1_model_path", type=str, required=True,
                   help="Path to Stage 1 VIB model directory")
parser.add_argument("--stage2_model_path", type=str, required=True,
                   help="Path to Stage 2 VIB model directory")

# Model hyperparameters (must match training)
parser.add_argument("--latent_dim", type=int, default=128,
                   help="Latent dimension for VIB")
parser.add_argument("--layer", type=str, default="all",
                   help="Layer index or 'all' for layer averaging")
parser.add_argument("--batch_size", type=int, default=4,
                   help="Batch size used during training")
parser.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Learning rate used during training")
parser.add_argument("--beta_s1", type=float, default=0.1,
                   help="Beta for stage 1")
parser.add_argument("--beta_s2", type=float, default=0.1,
                   help="Beta for stage 2")
parser.add_argument("--no_ib", action='store_true',
                   help="Whether information bottleneck was disabled during training")

# Generation parameters
parser.add_argument("--prompt", type=str, default="",
                   help="Text prompt for generation")
parser.add_argument("--gen_length", type=int, default=100,
                   help="Number of tokens to generate")
parser.add_argument("--temperature", type=float, default=0.7,
                   help="Temperature for sampling")
parser.add_argument("--top_k", type=int, default=100,
                   help="Top-k sampling parameter")
parser.add_argument("--top_p", type=float, default=0.9,
                   help="Top-p (nucleus) sampling parameter")
parser.add_argument("--repetition_penalty", type=float, default=1.5,
                   help="Repetition penalty")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    print(f"Using device: {device}")
    print(f"Loading base model from: {args.base_model_path}")
    print(f"Loading Stage 1 VIB from: {args.stage1_model_path}")
    print(f"Loading Stage 2 VIB from: {args.stage2_model_path}")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = LlamaConfig.from_pretrained(args.base_model_path)
    preLM = LlamaModel.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    ).eval()
    
    print("Base model loaded successfully")

    # Load Stage 1 VIB model
    layer_weight_averaging = (args.layer == 'all')
    stage1_config = VIBConfig(
        input_dim=config.hidden_size,
        latent_dim=args.latent_dim,
        stage="1",
        num_classes=2,  # Binary PII classification
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
    
    stage1_model_file = os.path.join(args.stage1_model_path, f"model{s1_postfix}.pth")
    
    if not os.path.exists(stage1_model_file):
        print(f"Stage 1 model not found: {stage1_model_file}")
        import glob
        alt_s1_models = glob.glob(os.path.join(args.stage1_model_path, "model*.pth"))
        if alt_s1_models:
            print(f"Found alternative Stage 1 models: {alt_s1_models}")
            stage1_model_file = alt_s1_models[0]
            print(f"Using: {stage1_model_file}")
        else:
            raise FileNotFoundError(f"No Stage 1 model found in {args.stage1_model_path}")
    
    model_device = next(preLM.parameters()).device if hasattr(preLM, 'parameters') else device
    stage1_vib.load_state_dict(torch.load(stage1_model_file, map_location=model_device))
    stage1_vib.eval()
    stage1_vib.to(model_device)
    print("Stage 1 VIB loaded successfully")

    # Load Stage 2 VIB model
    stage2_config = VIBConfig(
        input_dim=config.hidden_size,
        latent_dim=args.latent_dim,
        stage="2",
        num_classes=tokenizer.vocab_size,  # Language modeling
        layer_weight_averaging=layer_weight_averaging,
        num_layers=config.num_hidden_layers if layer_weight_averaging else None
    )
    stage2_vib = VIB(stage2_config)

    # Construct Stage 2 model filename
    s2_postfix = f"_bs={args.batch_size}_lr={args.learning_rate}_dim={args.latent_dim}"
    if args.no_ib:
        s2_postfix += "_noib"
    else:
        s2_postfix += f"_b={args.beta_s1}_{args.beta_s2}"
    s2_postfix += f"_layer={args.layer}"
    
    stage2_model_file = os.path.join(args.stage2_model_path, f"model{s2_postfix}.pth")
    
    if not os.path.exists(stage2_model_file):
        print(f"Stage 2 model not found: {stage2_model_file}")
        import glob
        alt_s2_models = glob.glob(os.path.join(args.stage2_model_path, "model*.pth"))
        if alt_s2_models:
            print(f"Found alternative Stage 2 models: {alt_s2_models}")
            stage2_model_file = alt_s2_models[0]
            print(f"Using: {stage2_model_file}")
        else:
            raise FileNotFoundError(f"No Stage 2 model found in {args.stage2_model_path}")
    
    stage2_vib.load_state_dict(torch.load(stage2_model_file, map_location=model_device))
    stage2_vib.eval()
    stage2_vib.to(model_device)
    print("Stage 2 VIB loaded successfully")

    # Prepare input
    if args.prompt:
        print(f"\nPrompt: '{args.prompt}'")
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(model_device)
    else:
        print("\nGenerating from empty prompt...")
        input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(model_device)

    print(f"Generating {args.gen_length} tokens...")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    
    start_time = time.time()
    
    # Get Stage 1 conditioning for the initial prompt
    with torch.no_grad():
        # Get hidden states from base model for conditioning
        outputs = preLM(input_ids, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)
        hidden_states = hidden_states[1:].permute(1, 0, 2, 3)
        
        # Get Stage 1 conditioning
        _, stage1_cond, _ = stage1_vib(hidden_states, noise=False)
        
        # Generate using VIB's generate method
        generated_ids, final_outputs = stage2_vib.generate(
            ids=input_ids,
            preLM=preLM,
            cond=stage1_cond,
            mask=None,
            length=args.gen_length,
            temp=args.temperature,
            topk=args.top_k,
            topp=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    # Decode and print
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated_text)
    print("="*80)
    print(f"\nGeneration time: {generation_time:.2f}s ({len(generated_ids[0])} tokens, {len(generated_ids[0])/generation_time:.2f} tokens/s)")
    print(f"Total tokens: {len(generated_ids[0])}")
