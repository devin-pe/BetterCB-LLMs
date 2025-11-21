from pathlib import Path
import argparse
import os
import glob
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
import numpy as np

from evaluate import load as load_metric

from transformers import AutoTokenizer, LlamaModel

from modules import VIB, VIBConfig


def load_echr_data(data_path, split, max_length):
    data = pd.read_csv(os.path.join(data_path, f"echr_{split}.csv"))
    dataset = Dataset.from_dict({
        'text': data['fact'].tolist(),
        'position': data['position'].tolist()
    })
    return dataset


def prepare_dataset_stage1(batch, tokenizer, max_length):
    tokenized = tokenizer(
        batch['text'],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    batch['input_ids'] = tokenized['input_ids']
    batch['attention_mask'] = tokenized['attention_mask']

    position_labels = []
    for pos_str in batch['position']:
        labels = [int(x) for x in pos_str.split(',')]
        if len(labels) < max_length:
            labels = labels + [0] * (max_length - len(labels))
        else:
            labels = labels[:max_length]
        position_labels.append(labels)
    batch['labels'] = position_labels
    return batch


def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
    attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
    labels = torch.tensor([x['labels'] for x in batch])
    return { 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels }


def find_checkpoint(root):
    paths = glob.glob(os.path.join(root, '**', 'model*.pth'), recursive=True)
    return paths[0] if paths else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/dpereira/CB-LLMs/analysing_pii_leakage/examples/experiments/experiment_00015')
    parser.add_argument('--stage1_ckpt', type=str, default=None)
    parser.add_argument('--stage2_ckpt', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='/home/dpereira/CB-LLMs/generation/dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu') if args.no_cuda or not torch.cuda.is_available() else torch.device('cuda:0')

    print(f'Loading tokenizer and base model from: {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = LlamaModel.from_pretrained(args.model_path)
    base_model.to(device)
    base_model.eval()
  
    stage1_ckpt = args.stage1_ckpt or find_checkpoint(root='./models/vib/1')
    if stage1_ckpt is None:
        raise RuntimeError('No Stage-1 checkpoint found (pass --stage1_ckpt)')
    stage2_ckpt = args.stage2_ckpt or find_checkpoint(root='./models/vib/0.01_2')
    if stage2_ckpt is None:
        raise RuntimeError('No Stage-2 checkpoint found (pass --stage2_ckpt)')
      
    print(f'Loading Stage-1 checkpoint: {stage1_ckpt}')
    stage1_checkpoint = torch.load(stage1_ckpt, map_location=device)
    stage1_latent_dim = stage1_checkpoint['encoder.mu.weight'].shape[0]
    stage1_has_layer_weights = 'layer_weights' in stage1_checkpoint
    print(f'Inferred Stage-1 latent dim: {stage1_latent_dim}')
    print(f'Stage-1 layer_weight_averaging: {stage1_has_layer_weights}')
    
    print(f'Loading Stage-2 checkpoint: {stage2_ckpt}')
    stage2_checkpoint = torch.load(stage2_ckpt, map_location=device)
    stage2_latent_dim = stage2_checkpoint['encoder.mu.weight'].shape[0]
    stage2_has_layer_weights = 'layer_weights' in stage2_checkpoint
    
    # Infer cond_dim from checkpoint if it has cond_projection
    stage2_cond_dim = None
    if 'decoder.cond_projection.weight' in stage2_checkpoint:
        stage2_cond_dim = stage2_checkpoint['decoder.cond_projection.weight'].shape[1]
        print(f'Inferred Stage-2 cond_dim: {stage2_cond_dim}')
    
    print(f'Inferred Stage-2 latent dim: {stage2_latent_dim}')
    print(f'Stage-2 layer_weight_averaging: {stage2_has_layer_weights}')

    # Build Stage-1 VIB model (for its decoder)
    stage1_config = VIBConfig(
        input_dim=base_model.config.hidden_size,
        latent_dim=stage1_latent_dim,
        stage='1',
        num_classes=2,
        layer_weight_averaging=stage1_has_layer_weights,
        num_layers=base_model.config.num_hidden_layers if stage1_has_layer_weights else None,
        cond_dim=None
    )
    stage1_vib = VIB(stage1_config)
    stage1_vib.load_state_dict(stage1_checkpoint)
    stage1_vib.to(device)
    stage1_vib.eval()
    print('Loaded Stage-1 VIB model (will use its decoder)')

    # Build Stage-2 VIB model (for extracting latents)
    stage2_config = VIBConfig(
        input_dim=base_model.config.hidden_size,
        latent_dim=stage2_latent_dim,
        stage='2',
        num_classes=tokenizer.vocab_size,
        layer_weight_averaging=stage2_has_layer_weights,
        num_layers=base_model.config.num_hidden_layers if stage2_has_layer_weights else None,
        cond_dim=stage2_cond_dim
    )
    stage2_vib = VIB(stage2_config)
    stage2_vib.load_state_dict(stage2_checkpoint)
    stage2_vib.to(device)
    stage2_vib.eval()
    print('Loaded Stage-2 VIB model (will use its encoder/latents)')

    # Create a projection layer if latent dimensions don't match
    if stage1_latent_dim != stage2_latent_dim:
        print(f'Stage-1 and Stage-2 latent dims differ ({stage1_latent_dim} vs {stage2_latent_dim})')
        print('Creating a linear projection layer')
        projection = torch.nn.Linear(stage2_latent_dim, stage1_latent_dim).to(device)
        projection.eval()
    else:
        print('Stage-1 and Stage-2 latent dims match, no projection needed')
        projection = None

    # Load and prepare test dataset (Stage 1 labels)
    print('Loading test dataset...')
    test_ds = load_echr_data(args.data_path, 'validation', args.max_length)
    test_ds = test_ds.map(lambda b: prepare_dataset_stage1(b, tokenizer, args.max_length), batched=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Evaluate both Stage-1 and Stage-2 latents
    print('Evaluating Stage-1 and Stage-2 latents with Stage-1 decoder on test labels...')
    
    # Accumulators for per-class accuracy
    s1_all_preds = []
    s1_all_refs = []
    s2_all_preds = []
    s2_all_refs = []
    
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = base_model(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True, return_dict=True)
            hidden_states = torch.stack(outputs.hidden_states)[1:].permute(1, 0, 2, 3)
            
            # Extract Stage-1 latents
            h1 = hidden_states
            if stage1_config.layer_weight_averaging:
                w1 = torch.nn.functional.softmax(stage1_vib.layer_weights, dim=0)
                h1 = torch.sum(h1 * w1.view(1, w1.shape[0], 1, 1), dim=1)
            mu1, _ = stage1_vib.encoder(h1)
            
            # Get Stage-1 predictions
            decoder_outputs_s1 = stage1_vib.decoder(mu1)
            logits_s1 = decoder_outputs_s1[0]
            
            # Extract Stage-2 latents
            h2 = hidden_states
            if stage2_config.layer_weight_averaging:
                w2 = torch.nn.functional.softmax(stage2_vib.layer_weights, dim=0)
                h2 = torch.sum(h2 * w2.view(1, w2.shape[0], 1, 1), dim=1)
            mu2, _ = stage2_vib.encoder(h2)
            
            # Project Stage-2 latents to Stage-1 dimension if needed
            if projection is not None:
                batch_size, seq_len, _ = mu2.shape
                mu2_flat = mu2.view(batch_size * seq_len, stage2_latent_dim)
                mu1_proj = projection(mu2_flat).view(batch_size, seq_len, stage1_latent_dim)
            else:
                mu1_proj = mu2
            
            # Get Stage-2 predictions using Stage-1 decoder
            decoder_outputs_s2 = stage1_vib.decoder(mu1_proj)
            logits_s2 = decoder_outputs_s2[0]

        # Process both Stage-1 and Stage-2 predictions
        batch_size, seq_len = batch['labels'].shape
        flat_labels = batch['labels'].view(batch_size * seq_len)
        flat_mask = batch['attention_mask'].view(batch_size * seq_len).bool()

        if flat_mask.sum() == 0:
            continue

        # Stage-1 predictions
        flat_logits_s1 = logits_s1.view(batch_size * seq_len, 2)
        filtered_logits_s1 = flat_logits_s1[flat_mask]
        preds_s1 = torch.argmax(filtered_logits_s1, dim=-1).cpu().numpy()
        
        # Stage-2 predictions
        flat_logits_s2 = logits_s2.view(batch_size * seq_len, 2)
        filtered_logits_s2 = flat_logits_s2[flat_mask]
        preds_s2 = torch.argmax(filtered_logits_s2, dim=-1).cpu().numpy()
        
        # References
        filtered_labels = flat_labels[flat_mask].cpu().numpy()
        
        # Accumulate
        s1_all_preds.extend(preds_s1)
        s1_all_refs.extend(filtered_labels)
        s2_all_preds.extend(preds_s2)
        s2_all_refs.extend(filtered_labels)

    # Convert to numpy arrays
    s1_all_preds = np.array(s1_all_preds)
    s1_all_refs = np.array(s1_all_refs)
    s2_all_preds = np.array(s2_all_preds)
    s2_all_refs = np.array(s2_all_refs)
    
    # Compute overall accuracies
    s1_acc = (s1_all_preds == s1_all_refs).mean()
    s2_acc = (s2_all_preds == s2_all_refs).mean()
    
    # Compute per-class accuracies
    s1_class0_mask = s1_all_refs == 0
    s1_class1_mask = s1_all_refs == 1
    s2_class0_mask = s2_all_refs == 0
    s2_class1_mask = s2_all_refs == 1
    
    s1_acc_class0 = (s1_all_preds[s1_class0_mask] == s1_all_refs[s1_class0_mask]).mean() if s1_class0_mask.sum() > 0 else 0.0
    s1_acc_class1 = (s1_all_preds[s1_class1_mask] == s1_all_refs[s1_class1_mask]).mean() if s1_class1_mask.sum() > 0 else 0.0
    s2_acc_class0 = (s2_all_preds[s2_class0_mask] == s2_all_refs[s2_class0_mask]).mean() if s2_class0_mask.sum() > 0 else 0.0
    s2_acc_class1 = (s2_all_preds[s2_class1_mask] == s2_all_refs[s2_class1_mask]).mean() if s2_class1_mask.sum() > 0 else 0.0
    
    # Print results
    print("\n" + "="*60)
    print("STAGE-1 LATENTS (with Stage-1 decoder):")
    print("="*60)
    print(f"Overall Test Accuracy: {s1_acc:.4f}")
    print(f"Class 0 (non-PII) Accuracy: {s1_acc_class0:.4f} ({s1_class0_mask.sum()} samples)")
    print(f"Class 1 (PII) Accuracy: {s1_acc_class1:.4f} ({s1_class1_mask.sum()} samples)")
    
    print("\n" + "="*60)
    print("STAGE-2 LATENTS (with Stage-1 decoder):")
    print("="*60)
    print(f"Overall Test Accuracy: {s2_acc:.4f}")
    print(f"Class 0 (non-PII) Accuracy: {s2_acc_class0:.4f} ({s2_class0_mask.sum()} samples)")
    print(f"Class 1 (PII) Accuracy: {s2_acc_class1:.4f} ({s2_class1_mask.sum()} samples)")
    print("="*60)


if __name__ == '__main__':
    main()
