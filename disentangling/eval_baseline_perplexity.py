"""
Evaluate fine-tuned LLaMA3 baseline model perplexity on ECHR test set.
Follows the same evaluation approach as training_stage2.py.
"""

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

from dataset_utils import load_echr_data


def prepare_dataset(batch, tokenizer, max_length=512):
    """Prepare batch for evaluation"""
    tokenized = tokenizer(
        batch["text"],
        padding=False,
        truncation=True,
        max_length=max_length
    )
    
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids']
    }


def create_collate_fn(tokenizer):
    """Collate function for batching"""
    def collate_fn(batch):
        input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
        attention_mask_list = [torch.tensor(x["attention_mask"]) for x in batch]
        labels_list = [torch.tensor(x["labels"]) for x in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    return collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                        default="/home/dpereira/CB-LLMs/analysing_pii_leakage/examples/experiments/experiment_00015",
                        help="Path to fine-tuned LLaMA3 model")
    parser.add_argument("--data_path", type=str, 
                        default="/home/dpereira/CB-LLMs/generation/dataset/",
                        help="Path to ECHR dataset")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (default: 512)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation (default: 4)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device to use (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Evaluating fine-tuned LLaMA3 baseline on ECHR test set")
    print(f"Model path: {args.model_path}")
    print(f"Max length: {args.max_length}")
    print(f"Batch size: {args.batch_size}")
    
    # GPU setup
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f'Using GPU: {torch.cuda.get_device_name(args.gpu)}')
    else:
        device = torch.device("cpu")
        print('No GPU available, using CPU')
    
    # Load tokenizer and model
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    print(f"Model loaded successfully")
    
    # Load test dataset
    print(f"\nLoading ECHR test dataset from {args.data_path}...")
    test_data_raw = load_echr_data('test', stage="2", data_path=args.data_path)
    
    # Prepare dataset
    test_data = test_data_raw.map(
        lambda batch: prepare_dataset(batch, tokenizer, args.max_length), 
        batched=True
    )
    
    # Create data loader
    collate_fn = create_collate_fn(tokenizer)
    test_dataloader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=0
    )
    
    print(f"Test dataset loaded: {len(test_data)} samples")
    print(f"Test batches: {len(test_dataloader)}")
    
    # Evaluate perplexity
    print(f"\nEvaluating perplexity...")
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            
            # Compute perplexity (same as training_stage2.py)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch['labels'][..., 1:].contiguous()
            
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Filter out padding tokens
            mask = shift_labels != -100
            if mask.sum() > 0:
                filtered_logits = shift_logits[mask]
                filtered_labels = shift_labels[mask]
                
                log_probs = F.log_softmax(filtered_logits, dim=-1)
                token_log_probs = log_probs.gather(1, filtered_labels.unsqueeze(1)).squeeze(1)
                
                predictions = token_log_probs.cpu().numpy().tolist()
                references = [1] * len(predictions)
                
                all_predictions.extend(predictions)
                all_references.extend(references)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_dataloader)} batches...")
    
    # Compute final perplexity
    if all_predictions:
        avg_log_prob = np.mean(all_predictions)
        perplexity = np.exp(-avg_log_prob)
        print(f"\n{'='*60}")
        print(f"Test Set Perplexity: {perplexity:.4f}")
        print(f"Average Log Probability: {avg_log_prob:.6f}")
        print(f"Total tokens evaluated: {len(all_predictions)}")
        print(f"{'='*60}")
    else:
        print("No valid predictions for perplexity calculation")


if __name__ == "__main__":
    main()
