"""
Stage 1 Training: PII Detection (token-level PERSON classification)
Trains a Variational Information Bottleneck (VIB) model to detect person entities in text.
"""
import argparse
import sys, os
import pickle
import numpy as np
import uuid
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset, load_from_disk
from evaluate import load
from transformers import AutoTokenizer, LlamaModel
from transformers import get_cosine_schedule_with_warmup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

from modules import VIB, VIBConfig
from dataset_utils import load_echr_data, prepare_dataset_stage1, create_collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA", type=str, default="custom_echr")
    parser.add_argument("--LATENT_DIM", type=int, default=4096)
    parser.add_argument("--MODEL_NAME", type=str, default="llama3")
    parser.add_argument("--LAYER", type=str, default="all")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
    parser.add_argument("--BETA", type=float, default=0.1)
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--NO_IB", action='store_true')
    parser.add_argument("--MAX_LENGTH", type=int, default=512)
    parser.add_argument("--BATCH_SIZE", type=int, default=4)
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--EVAL_FREQ", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Using dataset: {args.DATA}, Stage: 1")
    print(f"Task objective: position_labels (token-level PERSON detection)")
    
    # Training configuration
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.005
    SELECTED_GPU = 0
    
    # Paths
    DATA_PATH = "/home/dpereira/CB-LLMs/generation/dataset/"
    SAVE_REPORTS_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/reports/vib/4096_1/{args.DATA}/{args.MODEL_NAME}/"
    SAVE_MODEL_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/models/vib/4096_1/{args.DATA}/{args.MODEL_NAME}/"
    
    print(f"Model will be saved at {SAVE_MODEL_PATH}")
    
    if not os.path.exists(SAVE_REPORTS_PATH):
        os.makedirs(SAVE_REPORTS_PATH)
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)
    
    # GPU setup
    if torch.cuda.is_available():     
        device = torch.device(f"cuda:{SELECTED_GPU}")
        print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')
    
    # Load tokenizer and base model
    model_path = "/home/dpereira/CB-LLMs/analysing_pii_leakage/examples/experiments/experiment_00015"
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
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
    
    # Create VIB model for Stage 1
    layer = args.LAYER if args.LAYER == "all" else int(args.LAYER)
    layer_weight_averaging = (layer == "all")
    
    vib_config = VIBConfig(
        input_dim=base_model.config.hidden_size,
        latent_dim=args.LATENT_DIM,
        stage="1",
        num_classes=2,  # Binary classification for PERSON detection
        layer_weight_averaging=layer_weight_averaging,
        num_layers=base_model.config.num_hidden_layers if layer_weight_averaging else None,
        cond_dim=None
    )
    model = VIB(vib_config)
    model.to(device)
    model.train()
    
    # Load data
    print("Loading datasets...")
    train_data = load_echr_data('train', stage="1", data_path=DATA_PATH)
    test_data = load_echr_data('test', stage="1", data_path=DATA_PATH)
    
    # Prepare datasets
    train_data = train_data.map(lambda batch: prepare_dataset_stage1(batch, tokenizer, args.MAX_LENGTH), batched=True)
    test_data = test_data.map(lambda batch: prepare_dataset_stage1(batch, tokenizer, args.MAX_LENGTH), batched=True)
    
    # Create data loaders
    collate_fn = create_collate_fn(stage="1", tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=args.BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_data, 
        batch_size=args.BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=0
    )
    
    training_steps = len(train_dataloader)
    total_training_steps = args.EPOCHS * training_steps
    
    print(f"Training steps per epoch: {training_steps}")
    
    # Print run configuration
    print("\n" + "="*80)
    print("STAGE 1 TRAINING CONFIGURATION")
    print("="*80)
    print(f"Total Epochs: {args.EPOCHS}")
    print(f"Batch Size: {args.BATCH_SIZE}")
    print(f"Learning Rate: {args.LEARNING_RATE}")
    print(f"Latent Dimension: {args.LATENT_DIM}")
    print(f"Beta: {args.BETA}")
    print(f"Layer: {args.LAYER}")
    print(f"Model save location: {SAVE_MODEL_PATH}")
    print("="*80 + "\n")
    
    # Optimizer and scheduler
    metric = load('accuracy', experiment_id=str(uuid.uuid4()))
    optimizer = AdamW(params=model.parameters(), lr=args.LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(WARMUP_RATIO * total_training_steps), 
        num_training_steps=total_training_steps
    )
    
    beta = float(args.BETA)
    
    # Training loop
    train_losses = {'Task': [], 'Info': [], 'Total': []}
    test_performances = []
    best_total_loss = float('inf')
    best_model_state = None
    
    for epoch in range(args.EPOCHS):
        model.train()
        epoch_task_loss = 0
        epoch_info_loss = 0
        epoch_total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Feature extraction from pre-trained language model
            with torch.no_grad():
                outputs = base_model(
                    batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True
                )
            
            hidden_states = torch.stack(outputs.hidden_states)
            # Transform to batch-first and skip embedding layer
            hidden_states = hidden_states[1:].permute(1, 0, 2, 3)  # (batch, layers, seq, hidden)
            hidden_states = hidden_states.float()
            
            # Clear cache after base model forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Forward VIB model
            logits, mu, var = model(
                hidden_states if layer == "all" else hidden_states[:, layer:layer+1],
                m=batch["attention_mask"], 
                noise=not args.NO_IB
            )
            
            # Compute information loss
            if args.NO_IB:
                info_loss = torch.tensor(0.0, device=device)
            else:
                # Stage 1: KL divergence loss (per-token)
                info_loss = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=-1)
                # Apply mask for Stage 1 (sequence-level loss)
                info_loss = torch.masked_select(info_loss, batch["attention_mask"].bool()).mean()
            
            # Compute task loss (token-level sequence labeling)
            batch_size, seq_len, num_classes = logits.shape
            
            # Flatten for loss calculation
            flat_logits = logits.view(batch_size * seq_len, num_classes)  # [batch*seq, 2]
            flat_labels = batch['labels'].view(batch_size * seq_len)  # [batch*seq]
            
            # Create mask to ignore padding tokens
            flat_mask = batch["attention_mask"].view(batch_size * seq_len).bool()
            
            # Compute loss only on non-padded tokens
            task_loss = F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])
            
            # Total loss
            total_loss = task_loss + beta * info_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Track losses
            epoch_task_loss += task_loss.item()
            epoch_info_loss += info_loss.item() if not args.NO_IB else 0.0
            epoch_total_loss += total_loss.item()
        
        # Compute average losses
        avg_total_loss = epoch_total_loss / len(train_dataloader)
        avg_task_loss = epoch_task_loss / len(train_dataloader)
        avg_info_loss = epoch_info_loss / len(train_dataloader) if not args.NO_IB else 0.0
        
        train_losses['Task'].append(avg_task_loss)
        if not args.NO_IB:
            train_losses['Info'].append(avg_info_loss)
            train_losses['Total'].append(avg_total_loss)
        
        # Save best model
        if avg_total_loss < best_total_loss:
            best_total_loss = avg_total_loss
            best_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1}/{args.EPOCHS}, Task Loss: {avg_task_loss:.4f}, "
                  f"Info Loss: {avg_info_loss:.4f}, Total Loss: {avg_total_loss:.4f} -> New best!")
        else:
            print(f"Epoch {epoch+1}/{args.EPOCHS}, Task Loss: {avg_task_loss:.4f}, "
                  f"Info Loss: {avg_info_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % args.EVAL_FREQ == 0:
            model.eval()
            
            for batch in test_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with torch.no_grad():
                    # Feature extraction
                    outputs = base_model(
                        batch["input_ids"], 
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                hidden_states = torch.stack(outputs.hidden_states)
                hidden_states = hidden_states[1:].permute(1, 0, 2, 3)
                hidden_states = hidden_states.float()
                
                # Forward VIB model
                with torch.no_grad():
                    logits, mu, var = model(
                        hidden_states if layer == "all" else hidden_states[:, layer:layer+1],
                        m=batch["attention_mask"]
                    )
                
                # Compute predictions
                preds = torch.argmax(logits, dim=-1)  # [batch, seq_len]
                batch_size, seq_len = preds.shape
                flat_preds = preds.view(batch_size * seq_len)
                flat_labels = batch['labels'].view(batch_size * seq_len)
                flat_mask = batch["attention_mask"].view(batch_size * seq_len).bool()
                
                # Only evaluate on non-padded tokens
                predictions = flat_preds[flat_mask].cpu().numpy()
                references = flat_labels[flat_mask].cpu().numpy()
                
                metric.add_batch(predictions=predictions, references=references)
            
            # Compute accuracy
            perf = metric.compute()['accuracy']
            print(f"  Test Accuracy: {perf:.4f}")
            test_performances.append(perf)
            
            model.train()
    
    # Save results
    postfix = f"_bs={args.BATCH_SIZE}_lr={args.LEARNING_RATE}_dim={args.LATENT_DIM}"
    if args.NO_IB:
        postfix += "_noib" 
    else:
        postfix += f"_b={args.BETA}"
    postfix += f"_layer={args.LAYER}"
    
    print(f"\nSaving results with postfix: {postfix}")
    
    with open(f"{SAVE_REPORTS_PATH}train_losses{postfix}.pkl", 'wb') as f:
        pickle.dump(train_losses, f)
    
    with open(f"{SAVE_REPORTS_PATH}test_accuracy{postfix}.pkl", 'wb') as f:
        pickle.dump(test_performances, f)
    
    # Save best model
    model_state_to_save = best_model_state if best_model_state is not None else model.state_dict()
    torch.save(model_state_to_save, f'{SAVE_MODEL_PATH}model{postfix}.pth')
    print(f"Saved best model with Total Loss: {best_total_loss:.4f}")
    
    if layer_weight_averaging:
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        layer_weights = torch.nn.functional.softmax(model.layer_weights, dim=0).detach().cpu().numpy().tolist()
        with open(f"{SAVE_MODEL_PATH}layer-weights{postfix}.pkl", 'wb') as f:
            pickle.dump(layer_weights, f)
    
    print(f"\nStage 1 training completed!")
    print(f"Best model saved with Total Loss: {best_total_loss:.4f}")
    print(f"Final test accuracy: {test_performances[-1] if test_performances else 'N/A'}")


if __name__ == "__main__":
    main()
