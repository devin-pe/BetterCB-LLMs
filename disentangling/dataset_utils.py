"""
Dataset utilities for CB-LLM training
Contains dataset loading and preparation functions for both Stage 1 and Stage 2
"""
import torch
import pandas as pd
from datasets import Dataset


def load_echr_data(split, stage, data_path="/home/dpereira/CB-LLMs/generation/dataset/"):
    """Load ECHR dataset for the specified split and stage"""
    data = pd.read_csv(f"{data_path}echr_{split}.csv")
    print(f"Loaded ECHR {split} dataset with {len(data)} samples")
    
    if stage == "1":
        # Stage 1: Token-level PERSON detection
        dataset = Dataset.from_dict({
            'text': data['fact'].tolist(),
            'position': data['position'].tolist()  # Comma-separated string of 0s and 1s
        })
    else:
        # Stage 2: Language modeling task (predict next word)
        texts = data['fact'].tolist()
        # For language modeling, we'll use the text as input
        # Labels will be created from tokenization in prepare_dataset_stage2
        dataset = Dataset.from_dict({
            'text': texts
        })
    
    return dataset


def prepare_dataset_stage1(batch, tokenizer, max_length=512):
    """Prepare batch for Stage 1 (PII classification or token-level detection)"""
    # Tokenize text
    tokenized = tokenizer(
        batch["text"], 
        padding=True, 
        truncation=True, 
        max_length=max_length,
        return_tensors="pt"
    )
    
    batch['input_ids'] = tokenized['input_ids']
    batch['attention_mask'] = tokenized['attention_mask']
    
    position_labels = []
    for pos_str in batch['position']:
        labels = [int(x) for x in pos_str.split(',')]
        # Pad or truncate to max_length to match tokenized sequence
        if len(labels) < max_length:
            labels = labels + [0] * (max_length - len(labels))
        else:
            labels = labels[:max_length]
        position_labels.append(labels)
    batch['labels'] = position_labels

    return batch


def prepare_dataset_stage2(batch, tokenizer, max_length=512):
    """Prepare batch for Stage 2 (language modeling)"""
    # Tokenize text for language modeling - no padding here, do it in collate_fn
    tokenized = tokenizer(
        batch["text"], 
        padding=False,  # Don't pad here, will pad in collate_fn
        truncation=True, 
        max_length=max_length
    )
    
    # Return lists, not tensors - DataLoader will handle batching
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids']  # For language modeling
    }


def create_collate_fn(stage, tokenizer):
    """Create collate function for DataLoader based on stage"""
    def collate_fn(batch):
        if stage == "1":
            # Stage 1: Classification
            input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
            attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
            labels = torch.tensor([x["labels"] for x in batch])
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # Stage 2: Language modeling
            # Handle variable-length sequences with proper padding
            input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
            attention_mask_list = [torch.tensor(x["attention_mask"]) for x in batch]
            labels_list = [torch.tensor(x["labels"]) for x in batch]
            
            # Pad sequences to same length in the batch
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
    
    return collate_fn
