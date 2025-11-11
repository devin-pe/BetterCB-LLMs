import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import LlamaForCausalLM, AutoTokenizer
import time

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--model_type", type=str, choices=["finetuned", "pretrained"], default="finetuned", 
                   help="Type of model to use: 'finetuned' (local experiment_00015) or 'pretrained' (HF Llama3-8B)")
parser.add_argument("--model_path", type=str, default="analysing_pii_leakage/examples/experiments/experiment_00015", 
                   help="Path to the fine-tuned model (used when model_type='finetuned')")
parser.add_argument("--gen_length", type=int, default=100, help="Number of tokens to generate")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling")
parser.add_argument("--prompt", type=str, default="", help="Input prompt for generation")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    print("Loading Llama model...")
    
    # Determine model path based on model type
    if args.model_type == "pretrained":
        model_path = "meta-llama/Meta-Llama-3-8B"
        print(f"Using pretrained Llama3-8B from Hugging Face: {model_path}")
    else:  # finetuned
        model_path = args.model_path
        print(f"Using fine-tuned model from: {model_path}")
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()

    print(f"Model loaded successfully!")
    print(f"Model type: {args.model_type}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Prepare input
    input_text = args.prompt
    print(f"Input prompt: '{input_text}'")
    
    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    print(f"Generating {args.gen_length} tokens...")
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.gen_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    
    # Decode and display results
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_text = generated_text[len(input_text):]  # Extract only the newly generated part
    
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    print("\n" + "="*50)
    print("NEW TOKENS ONLY:")
    print("="*50)
    print(new_text)
    print("\n" + "="*50)
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Tokens generated: {len(output_ids[0]) - len(input_ids[0])}")
    print(f"Generation speed: {len(output_ids[0]) - len(input_ids[0]) / generation_time:.2f} tokens/sec")