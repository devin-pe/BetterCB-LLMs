#!/usr/bin/env python3
"""
Add token-level position labels for PERSON entities to ECHR dataset.

This script:
1. Uses Flair NER to detect PERSON entities at the word level
2. Maps these to Llama tokenizer token positions
3. Adds a 'position' column with binary labels (1 for PERSON tokens, 0 otherwise)

The position labels can be used as targets for cross-entropy loss during training.
"""

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import AutoTokenizer
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_flair_tagger():
    """Load Flair NER tagger for detecting PERSON entities."""
    print("Loading Flair NER model...")
    tagger = SequenceTagger.load("flair/ner-english-large")
    return tagger


def detect_person_entities(text, tagger):
    """
    Detect PERSON entities in text using Flair.
    
    Returns:
        List of tuples (start_char, end_char, entity_text)
    """
    sentence = Sentence(text)
    tagger.predict(sentence)
    
    person_entities = []
    for entity in sentence.get_spans('ner'):
        if entity.tag == 'PER':  # PERSON entity
            person_entities.append({
                'start_char': entity.start_position,
                'end_char': entity.end_position,
                'text': entity.text
            })
    
    return person_entities


def map_to_token_positions(text, person_entities, tokenizer, max_length=512):
    """
    Map character-level PERSON entity positions to token-level positions.
    
    Args:
        text: Input text
        person_entities: List of dicts with 'start_char' and 'end_char'
        tokenizer: Llama tokenizer
        max_length: Maximum sequence length
    
    Returns:
        List of 0s and 1s (1 for PERSON tokens, 0 otherwise)
    """
    # Tokenize the text
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=True
    )
    
    token_positions = [0] * len(encoding['input_ids'])
    offset_mapping = encoding['offset_mapping']
    
    # For each PERSON entity, mark overlapping tokens
    for entity in person_entities:
        entity_start = entity['start_char']
        entity_end = entity['end_char']
        
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            # Skip special tokens (they have (0, 0) offsets)
            if token_start == 0 and token_end == 0 and idx > 0:
                continue
            
            # Check if token overlaps with entity span
            if token_start < entity_end and token_end > entity_start:
                token_positions[idx] = 1
    
    return token_positions


def process_dataset(input_path, output_path, tokenizer, tagger, max_length=512):
    """
    Process a dataset CSV file and add position labels.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        tokenizer: Llama tokenizer
        tagger: Flair NER tagger
        max_length: Maximum sequence length
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Processing {len(df)} samples...")
    
    position_labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = row['fact']
        
        # Detect PERSON entities with Flair
        person_entities = detect_person_entities(text, tagger)
        
        # Map to token positions
        token_positions = map_to_token_positions(text, person_entities, tokenizer, max_length)
        
        # Store as comma-separated string for CSV
        position_str = ','.join(map(str, token_positions))
        position_labels.append(position_str)
    
    # Add position column
    df['position'] = position_labels
    
    # Save updated dataset
    print(f"Saving updated dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Print statistics
    total_tokens = sum(len(pos.split(',')) for pos in position_labels)
    person_tokens = sum(pos.split(',').count('1') for pos in position_labels)
    
    print(f"\nStatistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  PERSON tokens: {person_tokens} ({100*person_tokens/total_tokens:.2f}%)")
    print(f"  Samples with PERSON: {sum(1 for pos in position_labels if '1' in pos)} ({100*sum(1 for pos in position_labels if '1' in pos)/len(df):.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Add token-level PERSON position labels to ECHR dataset"
    )
    parser.add_argument(
        '--base_model_path',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help='Path to base Llama model for tokenizer'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/home/dpereira/CB-LLMs/generation/dataset',
        help='Directory containing ECHR dataset CSVs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (defaults to overwriting input files)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length for tokenization'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'validation', 'test'],
        help='Dataset splits to process'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.dataset_dir
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = args.base_model_path
    print(f"Loading Llama tokenizer from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Load Flair NER tagger
    tagger = load_flair_tagger()
    
    # Process each split
    for split in args.splits:
        input_path = os.path.join(args.dataset_dir, f'echr_{split}.csv')
        output_path = os.path.join(output_dir, f'echr_{split}.csv')
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        
        process_dataset(
            input_path=input_path,
            output_path=output_path,
            tokenizer=tokenizer,
            tagger=tagger,
            max_length=args.max_length
        )
    
    print("\n" + "="*60)
    print("All splits processed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
