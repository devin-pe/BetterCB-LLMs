"""
Evaluate Baseline LLaMA3 Top-5 Predictions at PII Positions
Extracts top-5 most likely tokens at positions where position=1 (PII entities)
from a random subset of 1000 PII tokens in ECHR training data.
Uses the fine-tuned LLaMA3 baseline model from experiment_00015.
"""
import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from collections import defaultdict
from flair.data import Sentence
from flair.models import SequenceTagger

from dataset_utils import load_echr_data


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate baseline LLaMA3 top-5 predictions at PII positions')
    parser.add_argument('--model_path', type=str, 
                        default='/home/dpereira/CB-LLMs/analysing_pii_leakage/examples/experiments/experiment_00015',
                        help='Path to fine-tuned LLaMA3 model')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--num_pii_tokens', type=int, default=1000,
                        help='Number of PII tokens to sample')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to extract')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def load_model(args, device):
    """Load fine-tuned LLaMA3 baseline model"""
    print(f"Loading fine-tuned LLaMA3 model from {args.model_path}")
    
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
    print(f"Vocabulary size: {model.config.vocab_size}")
    
    return model, tokenizer


def prepare_tokenized_data(batch, tokenizer, max_length=512):
    """Tokenize and keep position labels"""
    tokenized = tokenizer(
        batch["text"],
        padding=False,
        truncation=True,
        max_length=max_length
    )
    
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'position': batch['position']
    }


def sample_pii_tokens(dataset, tokenizer, num_pii_tokens=1000, max_length=512, seed=42):
    """
    Sample a random subset with exactly num_pii_tokens PII tokens (position=1).
    Returns list of (sample_idx, token_positions) tuples where token_positions 
    are indices within the sequence where position=1.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Find all samples with PII and their PII token positions
    pii_candidates = []  # List of (sample_idx, [pii_token_indices])
    
    for idx, sample in enumerate(dataset):
        pos_str = sample['position']
        pos_labels = [int(x) for x in pos_str.split(',')]
        
        # Tokenize to get actual sequence length after truncation
        tokenized = tokenizer(
            sample['text'],
            padding=False,
            truncation=True,
            max_length=max_length
        )
        seq_len = len(tokenized['input_ids'])
        
        # Find positions where position=1 (PII entity tokens)
        pii_positions = [i for i in range(min(len(pos_labels), seq_len)) if pos_labels[i] == 1]
        
        if pii_positions:
            pii_candidates.append((idx, pii_positions))
    
    print(f"Found {len(pii_candidates)} samples with PII tokens")
    
    # Randomly sample until we have num_pii_tokens
    selected_samples = []  # List of (sample_idx, [selected_token_positions])
    total_pii_tokens = 0
    
    # Shuffle candidates
    np.random.shuffle(pii_candidates)
    
    for sample_idx, pii_positions in pii_candidates:
        if total_pii_tokens >= num_pii_tokens:
            break
        
        # How many PII tokens do we still need?
        remaining = num_pii_tokens - total_pii_tokens
        
        # Take up to 'remaining' PII tokens from this sample
        tokens_to_take = min(len(pii_positions), remaining)
        selected_positions = pii_positions[:tokens_to_take]
        
        selected_samples.append((sample_idx, selected_positions))
        total_pii_tokens += len(selected_positions)
    
    print(f"Selected {len(selected_samples)} samples containing {total_pii_tokens} PII tokens")
    return selected_samples


def is_person_entity(text, tagger):
    """
    Check if text is tagged as a Person entity by Flair NER.
    Returns True if any span in the text is tagged as PER/PERSON.
    """
    if not text or not text.strip():
        return False
    
    sentence = Sentence(text.strip())
    tagger.predict(sentence)
    
    for entity in sentence.get_spans('ner'):
        # Check if entity is Person (PER, PERSON, etc.)
        if entity.tag in ['PER', 'PERSON']:
            return True
    return False


def extract_top_k_predictions(model, tokenizer, dataset, selected_samples, top_k, max_length, device, tagger):
    """
    For each PII token position in selected_samples, extract top-k predictions
    and count how many are Person entities according to Flair NER.
    Returns counts and statistics.
    """
    total_predictions = 0
    person_entity_count = 0
    
    results = []  # Keep for detailed printing
    
    for sample_idx, pii_positions in selected_samples:
        sample = dataset[sample_idx]
        
        # Tokenize
        tokenized = tokenizer(
            sample['text'],
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        # Get ground truth tokens at PII positions
        ground_truth_tokens = input_ids[0, pii_positions].cpu().tolist()
        ground_truth_text = [tokenizer.decode(tid) for tid in ground_truth_tokens]
        
        with torch.no_grad():
            # Get logits from baseline LLaMA3 model
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            logits = outputs.logits  # [1, seq_len, vocab_size]
            
            # Extract predictions at PII positions
            for pos_idx, pii_pos in enumerate(pii_positions):
                if pii_pos >= logits.shape[1]:
                    continue  # Skip if position is out of bounds
                
                # Get logits at this position
                pos_logits = logits[0, pii_pos, :]  # [vocab_size]
                
                # Get top-k predictions
                top_k_probs, top_k_indices = torch.topk(F.softmax(pos_logits, dim=-1), k=top_k)
                
                # Decode tokens
                top_k_ids = top_k_indices.cpu().tolist()
                top_k_tokens = [tokenizer.decode(idx) for idx in top_k_ids]
                top_k_probs = top_k_probs.cpu().tolist()
                
                # Check which top-k tokens are Person entities using Flair NER
                person_flags = []
                for token_text in top_k_tokens:
                    is_person = is_person_entity(token_text, tagger)
                    person_flags.append(is_person)
                    if is_person:
                        person_entity_count += 1
                
                total_predictions += len(top_k_tokens)
                
                # Check if ground truth is in top-k
                gt_token_id = ground_truth_tokens[pos_idx]
                gt_in_topk = gt_token_id in top_k_ids
                gt_rank = top_k_ids.index(gt_token_id) + 1 if gt_in_topk else -1
                
                # Get ground truth probability
                gt_prob = F.softmax(pos_logits, dim=-1)[gt_token_id].item()
                
                result = {
                    'sample_idx': sample_idx,
                    'position': pii_pos,
                    'ground_truth_id': gt_token_id,
                    'ground_truth_text': ground_truth_text[pos_idx],
                    'ground_truth_prob': gt_prob,
                    'gt_in_topk': gt_in_topk,
                    'gt_rank': gt_rank,
                    'top_k_ids': top_k_ids,
                    'top_k_tokens': top_k_tokens,
                    'top_k_probs': top_k_probs,
                    'person_flags': person_flags
                }
                
                results.append(result)
    
    return results, total_predictions, person_entity_count


def main():
    args = parse_args()
    
    print("="*80)
    print("BASELINE LLAMA3 TOP-K PREDICTION EVALUATION AT PII POSITIONS")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Number of PII tokens to sample: {args.num_pii_tokens}")
    print(f"Top-k predictions: {args.top_k}")
    print(f"Random seed: {args.seed}")
    print("="*80 + "\n")
    
    # Load Flair NER tagger
    print("Loading Flair NER tagger...")
    tagger = SequenceTagger.load('ner')
    print("Flair NER tagger loaded\n")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f'Using GPU: {torch.cuda.get_device_name(args.gpu)}')
    else:
        device = torch.device("cpu")
        print('Using CPU')
    
    # Load model
    model, tokenizer = load_model(args, device)
    
    # Load training dataset
    print("\nLoading ECHR training dataset...")
    DATA_PATH = "/home/dpereira/CB-LLMs/generation/dataset/"
    train_data = load_echr_data('train', stage="1", data_path=DATA_PATH)
    print(f"Loaded {len(train_data)} training samples")
    
    # Sample PII tokens
    print(f"\nSampling {args.num_pii_tokens} PII tokens from training data...")
    selected_samples = sample_pii_tokens(
        train_data, 
        tokenizer, 
        num_pii_tokens=args.num_pii_tokens,
        max_length=args.max_length,
        seed=args.seed
    )
    
    # Extract top-k predictions at PII positions and count Person entities
    print(f"\nExtracting top-{args.top_k} predictions at PII positions and checking for Person entities...")
    results, total_predictions, person_entity_count = extract_top_k_predictions(
        model, tokenizer, train_data, selected_samples, args.top_k, args.max_length, device, tagger
    )
    
    print(f"\nExtracted predictions for {len(results)} PII tokens")
    print(f"Total top-{args.top_k} predictions evaluated: {total_predictions}")
    
    # Compute statistics
    gt_in_topk_count = sum(1 for r in results if r['gt_in_topk'])
    gt_in_topk_rate = gt_in_topk_count / len(results) if results else 0
    avg_gt_prob = np.mean([r['ground_truth_prob'] for r in results]) if results else 0
    
    ranks = [r['gt_rank'] for r in results if r['gt_rank'] > 0]
    avg_rank = np.mean(ranks) if ranks else -1
    
    # Calculate Person entity percentage
    person_percentage = (person_entity_count / total_predictions * 100) if total_predictions > 0 else 0
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total PII tokens evaluated: {len(results)}")
    print(f"Total top-{args.top_k} predictions: {total_predictions}")
    print(f"\nPERSON ENTITY ANALYSIS (Flair NER):")
    print(f"  Person entities in top-{args.top_k} predictions: {person_entity_count}")
    print(f"  Percentage of Person entities: {person_percentage:.2f}%")
    print(f"\nGROUND TRUTH ANALYSIS:")
    print(f"  Ground truth in top-{args.top_k}: {gt_in_topk_count} ({100*gt_in_topk_rate:.2f}%)")
    print(f"  Average ground truth probability: {avg_gt_prob:.4f}")
    if ranks:
        print(f"  Average rank of ground truth (when in top-{args.top_k}): {avg_rank:.2f}")
    print("="*80)
    
    # Show first 10 examples
    print("\nFirst 10 examples:")
    print("-"*80)
    for i, result in enumerate(results[:10]):
        print(f"\nExample {i+1}:")
        print(f"  Sample: {result['sample_idx']}, Position: {result['position']}")
        print(f"  Ground truth: '{result['ground_truth_text']}' (ID: {result['ground_truth_id']}, Prob: {result['ground_truth_prob']:.4f})")
        print(f"  In top-{args.top_k}: {result['gt_in_topk']}, Rank: {result['gt_rank']}")
        print(f"  Top-{args.top_k} predictions:")
        for j, (token, prob, tid, is_person) in enumerate(zip(result['top_k_tokens'], result['top_k_probs'], result['top_k_ids'], result['person_flags'])):
            marker = " <-- GT" if tid == result['ground_truth_id'] else ""
            person_marker = " [PERSON]" if is_person else ""
            print(f"    {j+1}. '{token}' (ID: {tid}, Prob: {prob:.4f}){marker}{person_marker}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
