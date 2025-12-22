"""
Linear Probe for Stage 2 Latents
Evaluates whether PII information is still present in Stage 2 representations
by training a linear classifier on frozen Stage 2 latents.
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, LlamaModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

from modules import VIB, VIBConfig
from dataset_utils import load_echr_data, create_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Probe Stage 2 latents for PII leakage')
    parser.add_argument('--stage1_ckpt', type=str, required=True,
                        help='Path to Stage 1 VIB checkpoint directory')
    parser.add_argument('--stage2_ckpt', type=str, required=True,
                        help='Path to Stage 2 VIB checkpoint directory')
    parser.add_argument('--layer_s1', type=str, default='all',
                        help='Layer selection for Stage 1 (all or layer number)')
    parser.add_argument('--layer_s2', type=str, default='all',
                        help='Layer selection for Stage 2 (all or layer number)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for data loading')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--probe_epochs', type=int, default=10,
                        help='Number of epochs to train linear probe')
    parser.add_argument('--probe_lr', type=float, default=1e-3,
                        help='Learning rate for probe training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    return parser.parse_args()


def load_models(args, device):
    """Load base model, Stage 1 VIB, and Stage 2 VIB"""
    # Load base LLaMA model
    model_path = "/home/dpereira/CB-LLMs/analysing_pii_leakage/examples/experiments/experiment_00015"
    print(f"Loading base LLaMA model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = LlamaModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    base_model.to(device)
    base_model.eval()
    
    # Load Stage 1 VIB
    print(f"\nLoading Stage 1 VIB from {args.stage1_ckpt}")
    import glob
    stage1_files = glob.glob(os.path.join(args.stage1_ckpt, "model*.pth"))
    if not stage1_files:
        raise FileNotFoundError(f"No Stage 1 model found in {args.stage1_ckpt}")
    
    stage1_checkpoint = torch.load(stage1_files[0], map_location=device)
    stage1_latent_dim = stage1_checkpoint['encoder.mu.weight'].shape[0]
    stage1_layer_averaging = 'layer_weights' in stage1_checkpoint
    
    layer_s1 = args.layer_s1 if args.layer_s1 == "all" else int(args.layer_s1)
    stage1_config = VIBConfig(
        input_dim=base_model.config.hidden_size,
        latent_dim=stage1_latent_dim,
        stage="1",
        num_classes=2,
        layer_weight_averaging=stage1_layer_averaging,
        num_layers=base_model.config.num_hidden_layers if stage1_layer_averaging else None
    )
    
    stage1_vib = VIB(stage1_config)
    stage1_vib.load_state_dict(stage1_checkpoint)
    stage1_vib.to(device)
    stage1_vib.eval()
    print(f"Stage 1 latent_dim: {stage1_latent_dim}")
    
    # Load Stage 2 VIB
    print(f"\nLoading Stage 2 VIB from {args.stage2_ckpt}")
    stage2_files = glob.glob(os.path.join(args.stage2_ckpt, "model*.pth"))
    if not stage2_files:
        raise FileNotFoundError(f"No Stage 2 model found in {args.stage2_ckpt}")
    
    stage2_checkpoint = torch.load(stage2_files[0], map_location=device)
    stage2_latent_dim = stage2_checkpoint['encoder.mu.weight'].shape[0]
    stage2_layer_averaging = 'layer_weights' in stage2_checkpoint
    
    # Infer cond_dim
    cond_dim = stage1_latent_dim
    if 'decoder.cond_projection.weight' in stage2_checkpoint:
        cond_dim = stage2_checkpoint['decoder.cond_projection.weight'].shape[1]
    
    layer_s2 = args.layer_s2 if args.layer_s2 == "all" else int(args.layer_s2)
    stage2_config = VIBConfig(
        input_dim=base_model.config.hidden_size,
        latent_dim=stage2_latent_dim,
        stage="2",
        num_classes=tokenizer.vocab_size,
        layer_weight_averaging=stage2_layer_averaging,
        num_layers=base_model.config.num_hidden_layers if stage2_layer_averaging else None,
        cond_dim=cond_dim
    )
    
    stage2_vib = VIB(stage2_config)
    stage2_vib.load_state_dict(stage2_checkpoint)
    stage2_vib.to(device)
    stage2_vib.eval()
    print(f"Stage 2 latent_dim: {stage2_latent_dim}")
    
    return base_model, stage1_vib, stage2_vib, tokenizer, layer_s1, layer_s2


def prepare_dataset_with_labels(batch, tokenizer, max_length=512):
    """Prepare dataset with has_person labels"""
    tokenized = tokenizer(
        batch["text"],
        padding=False,
        truncation=True,
        max_length=max_length
    )
    
    # Extract has_person label from position string (1 if any position is 1, else 0)
    has_person_labels = []
    for pos_str in batch['position']:
        labels = [int(x) for x in pos_str.split(',')]
        has_person = 1 if any(labels) else 0
        has_person_labels.append(has_person)
    
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'has_person': has_person_labels
    }


def collate_with_labels(batch, tokenizer):
    """Collate function that includes has_person labels"""
    input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask_list = [torch.tensor(x["attention_mask"]) for x in batch]
    has_person_list = [x["has_person"] for x in batch]
    
    # Pad sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    has_person = torch.tensor(has_person_list, dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "has_person": has_person
    }


@torch.no_grad()
def extract_latents(base_model, stage1_vib, stage2_vib, dataloader, layer_s1, layer_s2, device):
    """Extract frozen Stage 2 latents from dataset"""
    all_latents = []
    all_labels = []
    
    print("\nExtracting Stage 2 latents...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}/{len(dataloader)}")
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward through base model
        outputs = base_model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = torch.stack(outputs.hidden_states)
        hidden_states = hidden_states[1:].permute(1, 0, 2, 3)  # [batch, layers, seq, hidden]
        hidden_states = hidden_states.float()
        
        # Get Stage 1 conditioning
        _, mu1, _ = stage1_vib(
            hidden_states if layer_s1 == "all" else hidden_states[:, layer_s1:layer_s1+1],
            m=batch["attention_mask"],
            noise=False
        )
        
        # Get Stage 2 latents (mu2)
        outputs_vib = stage2_vib(
            hidden_states if layer_s2 == "all" else hidden_states[:, layer_s2:layer_s2+1],
            m=batch["attention_mask"],
            cond=mu1,
            noise=False
        )
        _, _, mu2, _ = outputs_vib
        
        # Pool mu2 across sequence dimension (mean pooling over valid tokens)
        attention_mask_expanded = batch["attention_mask"].unsqueeze(-1).float()
        pooled_mu2 = (mu2 * attention_mask_expanded).sum(dim=1) / attention_mask_expanded.sum(dim=1).clamp(min=1.0)
        
        all_latents.append(pooled_mu2.cpu())
        all_labels.append(batch["has_person"].cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Extracted {all_latents.shape[0]} latent vectors of dimension {all_latents.shape[1]}")
    print(f"Label distribution: {all_labels.sum().item()} positive, {(all_labels == 0).sum().item()} negative")
    
    return all_latents, all_labels


def train_probe(train_latents, train_labels, val_latents, val_labels, args, device):
    """Train linear probe on frozen latents"""
    latent_dim = train_latents.shape[1]
    probe = nn.Linear(latent_dim, 2).to(device)
    optimizer = Adam(probe.parameters(), lr=args.probe_lr)
    
    print(f"\nTraining linear probe for {args.probe_epochs} epochs")
    print(f"Train samples: {len(train_labels)}, Val samples: {len(val_labels)}")
    
    best_val_acc = 0.0
    best_probe_state = None
    
    for epoch in range(args.probe_epochs):
        probe.train()
        
        # Simple batch training (not using DataLoader since data is already in memory)
        batch_size = 64
        indices = torch.randperm(len(train_labels))
        epoch_loss = 0.0
        
        for i in range(0, len(train_labels), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_latents = train_latents[batch_indices].to(device)
            batch_labels = train_labels[batch_indices].to(device)
            
            logits = probe(batch_latents)
            loss = F.cross_entropy(logits, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_latents.to(device))
            val_preds = val_logits.argmax(dim=-1).cpu()
            val_acc = accuracy_score(val_labels, val_preds)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_probe_state = probe.state_dict().copy()
        
        avg_loss = epoch_loss / (len(train_labels) / batch_size)
        print(f"Epoch {epoch+1}/{args.probe_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best probe
    probe.load_state_dict(best_probe_state)
    return probe, best_val_acc


def evaluate_probe(probe, latents, labels, device, split_name="Test"):
    """Evaluate probe performance"""
    probe.eval()
    with torch.no_grad():
        logits = probe(latents.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    
    labels_np = labels.numpy()
    
    acc = accuracy_score(labels_np, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds, average='binary')
    cm = confusion_matrix(labels_np, preds)
    
    print(f"\n{split_name} Set Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Compute per-class accuracy
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Specificity (neg class): {specificity:.4f}")
    print(f"Sensitivity (pos class): {sensitivity:.4f}")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    base_model, stage1_vib, stage2_vib, tokenizer, layer_s1, layer_s2 = load_models(args, device)
    
    # Freeze all models
    for model in [base_model, stage1_vib, stage2_vib]:
        for param in model.parameters():
            param.requires_grad = False
    
    # Load datasets
    data_path = "/home/dpereira/CB-LLMs/generation/dataset/"
    print(f"\nLoading datasets from {data_path}")
    
    train_data = load_echr_data('train', stage="1", data_path=data_path)
    test_data = load_echr_data('test', stage="1", data_path=data_path)
    
    # Prepare datasets with labels
    train_data = train_data.map(
        lambda batch: prepare_dataset_with_labels(batch, tokenizer, args.max_length),
        batched=True
    )
    test_data = test_data.map(
        lambda batch: prepare_dataset_with_labels(batch, tokenizer, args.max_length),
        batched=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_with_labels(batch, tokenizer),
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_with_labels(batch, tokenizer),
        shuffle=False,
        num_workers=0
    )
    
    # Extract latents
    train_latents, train_labels = extract_latents(
        base_model, stage1_vib, stage2_vib, train_loader, layer_s1, layer_s2, device
    )
    test_latents, test_labels = extract_latents(
        base_model, stage1_vib, stage2_vib, test_loader, layer_s1, layer_s2, device
    )
    
    # Split train into train/val (80/20)
    n_train = int(0.8 * len(train_labels))
    indices = torch.randperm(len(train_labels))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_latents_split = train_latents[train_indices]
    train_labels_split = train_labels[train_indices]
    val_latents = train_latents[val_indices]
    val_labels = train_labels[val_indices]
    
    # Train probe
    probe, best_val_acc = train_probe(
        train_latents_split, train_labels_split,
        val_latents, val_labels,
        args, device
    )
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    # Evaluate on test set
    test_results = evaluate_probe(probe, test_latents, test_labels, device, "Test")
    
    # Save results
    results = {
        'val_acc': best_val_acc,
        'test_results': test_results,
        'args': vars(args)
    }
    
    import pickle
    output_path = os.path.join(args.stage2_ckpt, 'probe_results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    if test_results['accuracy'] < 0.55:
        print("EXCELLENT: Stage 2 latents contain minimal PII information")
        print("(Classifier performs barely better than random guessing)")
    elif test_results['accuracy'] < 0.65:
        print("GOOD: Stage 2 latents have weak PII signal")
        print("(Some PII leaked but significantly reduced)")
    elif test_results['accuracy'] < 0.75:
        print("MODERATE: Stage 2 latents retain noticeable PII information")
        print("(Consider increasing BETA_S2 for stronger adversarial training)")
    else:
        print("POOR: Stage 2 latents strongly encode PII")
        print("(Adversarial training may not be effective)")
    print("="*80)


if __name__ == "__main__":
    main()
