import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, Dataset
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBL
import config as CFG
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="models/4096/custom_echr")
parser.add_argument("--tokenizer_path", type=str, default="/home/dpereira/CB-LLMs/analysing_pii_leakage/examples/experiments/experiment_00015")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)

# Fixed to custom_echr only
DATASET = "custom_echr"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text):
        self.encoded_text = encoded_text

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encoded_text.items()}
        return t

    def __len__(self):
        return len(self.encoded_text['input_ids'])


def build_loaders(encoded_text, mode, batch_size, num_workers):
    dataset = ClassificationDataset(encoded_text)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    return dataloader


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    print("="*80)
    print(f"Loading CBLLM model from: {args.model_path}")
    print(f"Dataset: {DATASET}")
    print("="*80)

    # Load tokenizer and config from the specified path
    print(f"\nLoading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = LlamaConfig.from_pretrained(args.tokenizer_path)

    # Load test dataset (custom_echr only)
    print("\nLoading ECHR test dataset...")
    test_data = pd.read_csv("/home/dpereira/CB-LLMs/generation/dataset/echr_test.csv")
    print(f"Loaded ECHR test dataset with {len(test_data)} samples")
    print(f"Test has_person distribution: {test_data['has_person'].value_counts().to_dict()}")
    
    test_dataset = Dataset.from_dict({
        'text': test_data['fact'].tolist(),
        'label': test_data['has_person'].tolist()
    })
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Print label distribution
    test_labels = np.array(test_dataset['label'])
    print(f"Test label distribution: {np.bincount(test_labels)}")

    # Tokenize test dataset
    print("Tokenizing test dataset...")
    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(
            e[CFG.example_name[DATASET]], 
            padding=True, 
            truncation=True, 
            max_length=args.max_length
        ), 
        batched=True,
        batch_size=len(test_dataset)
    )
    encoded_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    # Create test loader
    test_loader = build_loaders(encoded_test_dataset, mode="test", 
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers)

    # Get concept set (custom_echr)
    concept_set = CFG.concepts_from_labels[DATASET]
    print(f"Concept set: {concept_set}")
    print(f"Number of concepts: {len(concept_set)}")

    # Load pretrained model with LoRA
    print("\nLoading base model with LoRA from tokenizer path...")
    lora_config = LoraConfig(
        r=8, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        bias="none", 
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    preLM = LlamaModel.from_pretrained(args.tokenizer_path, torch_dtype=torch.bfloat16)
    preLM = get_peft_model(preLM, lora_config)
    
    # Load trained LoRA weights if they exist
    lora_path = os.path.join(args.model_path, "llama3_epoch_2")
    if os.path.exists(lora_path):
        print(f"Loading LoRA weights from: {lora_path}")
        preLM.load_adapter(lora_path, adapter_name="default")
    else:
        print(f"Warning: LoRA weights not found at {lora_path}, using base model")
    
    preLM = preLM.to(device)
    preLM.eval()

    # Load CBL
    print("\nLoading CBL model...")
    unsup_dim = CFG.unsup_dim.get(DATASET, CFG.unsup_dim.get('default', config.hidden_size))
    cbl = CBL(config, len(concept_set), tokenizer, unsup_dim=unsup_dim).to(device)
    
    # Load CBL checkpoint
    cbl_path = os.path.join(args.model_path, "cbl_epoch_2.pt")
    if os.path.exists(cbl_path):
        print(f"Loading CBL weights from: {cbl_path}")
        cbl.load_state_dict(torch.load(cbl_path, map_location=device))
    else:
        raise FileNotFoundError(f"CBL checkpoint not found at {cbl_path}")
    
    cbl.eval()

    print("\n" + "="*80)
    print("Calculating perplexity on test dataset...")
    print("="*80)
    
    all_log_probs = []
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                
                # Create word labels (shifted input_ids with padding masked)
                word_label = torch.where(
                    attention_mask[:, 1:] == 0, 
                    tokenizer.pad_token_id, 
                    input_ids[:, 1:]
                )
                
                # Forward pass through preLM
                features = preLM(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                ).last_hidden_state
                
                # Forward through CBL
                concepts, unsup, vocabs = cbl(features.float())
                
                # Calculate perplexity using shifted logits
                shift_logits = vocabs[:, :-1, :].contiguous()
                shift_labels = word_label.contiguous()
                
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Filter out padding tokens
                mask = shift_labels != tokenizer.pad_token_id
                if mask.sum() > 0:
                    filtered_logits = shift_logits[mask]
                    filtered_labels = shift_labels[mask]
                    
                    log_probs = F.log_softmax(filtered_logits, dim=-1)
                    token_log_probs = log_probs.gather(1, filtered_labels.unsqueeze(1)).squeeze(1)
                    
                    all_log_probs.extend(token_log_probs.cpu().numpy().tolist())
                    total_tokens += len(token_log_probs)
                
                # Progress indicator
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} batches", end="\r")
                    
            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                continue
    
    # Calculate final perplexity
    if all_log_probs:
        avg_log_prob = np.mean(all_log_probs)
        perplexity = np.exp(-avg_log_prob)
        print("\n" + "="*80)
        print(f"Test Perplexity: {perplexity:.4f}")
        print(f"Average log probability: {avg_log_prob:.4f}")
        print(f"Total tokens evaluated: {total_tokens}")
        print("="*80)
        
        # Save results
        results = {
            'perplexity': float(perplexity),
            'avg_log_prob': float(avg_log_prob),
            'total_tokens': int(total_tokens),
            'model_path': args.model_path,
            'tokenizer_path': args.tokenizer_path,
            'dataset': DATASET
        }
        
        import json
        results_path = os.path.join(args.model_path, "test_perplexity_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    else:
        print("\nWarning: No valid predictions for perplexity calculation")
