"""
Stage 2 Training: Language Modeling with Stage 1 Conditioning
Trains a VIB model for language modeling while conditioning on Stage 1 PII representations.
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
from evaluate import load
from transformers import AutoTokenizer, LlamaModel
from transformers import get_cosine_schedule_with_warmup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

from modules import VIB, VIBConfig
from dataset_utils import load_echr_data, prepare_dataset_stage2, create_collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_S1", type=str, default="custom_echr")
    parser.add_argument("--DATA_S2", type=str, default="custom_echr")
    parser.add_argument("--LATENT_DIM", type=int, default=512)  # Will be overridden to match LLaMA3
    parser.add_argument("--MODEL_NAME", type=str, default="llama3")
    parser.add_argument("--LAYER_S1", type=str, default="all")
    parser.add_argument("--LAYER_S2", type=str, default="all")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
    parser.add_argument("--BETA_S1", type=float, default=0.1)
    parser.add_argument("--BETA_S2", type=float, default=0.0)
    parser.add_argument("--BETA_S2_MSE", type=float, default=2.5)
    parser.add_argument("--var_clamp_min", type=float, default=0.1,
                        help='Minimum variance clamp value used in the encoder (default: 0.1)')
    parser.add_argument("--var_clamp_max", type=float, default=10.0,
                        help='Maximum variance clamp value used in the encoder (default: 10.0)')
    parser.add_argument("--beta_s2_start", type=float, default=None,
                        help='Starting beta_s2 for annealing (default: use BETA_S2)')
    parser.add_argument("--beta_s2_end", type=float, default=None,
                        help='Ending beta_s2 for annealing (default: use BETA_S2)')
    parser.add_argument("--beta_s2_warmup_epochs", type=int, default=0,
                        help='Number of epochs to linearly anneal beta_s2 from start to end (0=no annealing)')
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--NO_IB", action='store_true')
    parser.add_argument("--MAX_LENGTH", type=int, default=512)
    parser.add_argument("--BATCH_SIZE", type=int, default=4)
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--EVAL_FREQ", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Using dataset: {args.DATA_S2}, Stage: 2")
    print(f"Task objective: next_word (language modeling)")
    
    # Training configuration
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.005
    SELECTED_GPU = 0
    
    # Paths
    DATA_PATH = "/home/dpereira/CB-LLMs/generation/dataset/"
    LOAD_STAGE1_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/models/vib/4096_1/{args.DATA_S1}/{args.MODEL_NAME}/"
    DATA_ = args.DATA_S1 + "_" + args.DATA_S2
    SAVE_REPORTS_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/reports/vib/0.05_2/{DATA_}/{args.MODEL_NAME}/"
    SAVE_MODEL_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/models/vib/0.05_2/{DATA_}/{args.MODEL_NAME}/"
    
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
    
    # Load Stage 1 VIB model
    postfix = f"_bs={args.BATCH_SIZE}_lr={args.LEARNING_RATE}_dim={args.LATENT_DIM}"
    if args.NO_IB:
        postfix += "_noib" 
    else:
        postfix += f"_b={args.BETA_S1}"
    postfix += f"_layer={args.LAYER_S1}"
    
    stage1_model_file = f'{LOAD_STAGE1_PATH}model{postfix}.pth'
    
    if not os.path.exists(stage1_model_file):
        print(f"Stage 1 model not found: {stage1_model_file}")
        import glob
        alt_models = glob.glob(os.path.join(LOAD_STAGE1_PATH, "model*.pth"))
        if alt_models:
            stage1_model_file = alt_models[0]
            print(f"Found alternative Stage 1 model: {stage1_model_file}")
            print(f"Using: {stage1_model_file}")
        else:
            print(f"No Stage 1 model found in {LOAD_STAGE1_PATH}")
            print("Please train Stage 1 first!")
            exit(1)
    
    checkpoint = torch.load(stage1_model_file, map_location=device)
    stage1_latent_dim = checkpoint['encoder.mu.weight'].shape[0]
    print(f"Inferred Stage 1 latent_dim from checkpoint: {stage1_latent_dim}")
    
    layer_s1 = args.LAYER_S1 if args.LAYER_S1 == "all" else int(args.LAYER_S1)
    stage1_config = VIBConfig(
        input_dim=base_model.config.hidden_size,
        latent_dim=stage1_latent_dim,  
        stage="1",
        num_classes=2, 
        layer_weight_averaging=layer_s1 == "all",
        num_layers=base_model.config.num_hidden_layers if layer_s1 == "all" else None
    )
    # Pass variance clamp settings to Stage 1 config as well (harmless if not used)
    stage1_config.var_clamp_min = args.var_clamp_min
    stage1_config.var_clamp_max = args.var_clamp_max
    stage1_vib = VIB(stage1_config)
    stage1_vib.load_state_dict(checkpoint)
    stage1_vib.to(device)
    stage1_vib.eval()
    print(f"Loaded Stage 1 VIB model successfully from {stage1_model_file}")
    
    # Create Stage 2 VIB model
    # Stage 2 latent_dim must match LLaMA3 hidden size for proper merging with Stage 1
    LATENT_DIM = base_model.config.hidden_size  # 4096 for LLaMA3
    print(f"Stage 1 latent_dim: {stage1_latent_dim}, Stage 2 latent_dim: {LATENT_DIM} (enforced to match LLaMA3 hidden size)")
    
    layer_s2 = args.LAYER_S2 if args.LAYER_S2 in ["all", None] else int(args.LAYER_S2)
    layer_weight_averaging = (layer_s2 == "all")
    
    vib_config = VIBConfig(
        input_dim=base_model.config.hidden_size,
        latent_dim=LATENT_DIM,
        stage="2",
        num_classes=tokenizer.vocab_size,
        layer_weight_averaging=layer_weight_averaging,
        num_layers=base_model.config.num_hidden_layers if layer_weight_averaging else None,
        cond_dim=stage1_latent_dim
    )
    # Configure variance clamp bounds for Stage 2 encoder
    vib_config.var_clamp_min = args.var_clamp_min
    vib_config.var_clamp_max = args.var_clamp_max
    model = VIB(vib_config)
    model.to(device)
    model.train()
    
    # Load data
    print("Loading datasets...")
    train_data = load_echr_data('train', stage="2", data_path=DATA_PATH)
    test_data = load_echr_data('test', stage="2", data_path=DATA_PATH)
    
    # Prepare datasets
    print("[DEBUG] About to map train_data...")
    train_data = train_data.map(lambda batch: prepare_dataset_stage2(batch, tokenizer, args.MAX_LENGTH), batched=True)
    print("[DEBUG] Train data mapped successfully")
    
    print("[DEBUG] About to map test_data...")
    test_data = test_data.map(lambda batch: prepare_dataset_stage2(batch, tokenizer, args.MAX_LENGTH), batched=True)
    print("[DEBUG] Test data mapped successfully")
    
    # Create data loaders
    print("[DEBUG] Creating collate_fn...")
    collate_fn = create_collate_fn(stage="2", tokenizer=tokenizer)
    collate_fn = create_collate_fn(stage="2", tokenizer=tokenizer)
    print("[DEBUG] Collate_fn created successfully")
    
    print("[DEBUG] Creating train_dataloader...")
    train_dataloader = DataLoader(
        train_data, 
        batch_size=args.BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=0
    )
    print("[DEBUG] Train dataloader created successfully")
    
    print("[DEBUG] Creating test_dataloader...")
    test_dataloader = DataLoader(
        test_data, 
        batch_size=args.BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=0
    )
    print("[DEBUG] Test dataloader created successfully")
    
    print("[DEBUG] Computing training_steps (this triggers dataset iteration)...")
    training_steps = len(train_dataloader)
    print(f"[DEBUG] Training steps computed: {training_steps}")
    total_training_steps = args.EPOCHS * training_steps
    
    print(f"Training steps per epoch: {training_steps}")
    
    # Print run configuration
    print("\n" + "="*80)
    print("STAGE 2 TRAINING CONFIGURATION")
    print("="*80)
    print(f"Total Epochs: {args.EPOCHS}")
    print(f"Batch Size: {args.BATCH_SIZE}")
    print(f"Learning Rate: {args.LEARNING_RATE}")
    print(f"Stage 1 Latent Dimension: {stage1_latent_dim}")
    print(f"Stage 2 Latent Dimension: {LATENT_DIM}")
    print(f"Beta (Info Loss): {args.BETA_S2}")
    print(f"Beta MSE: {args.BETA_S2_MSE}")
    print(f"Layer S1: {args.LAYER_S1}, Layer S2: {args.LAYER_S2}")
    print(f"Model save location: {SAVE_MODEL_PATH}")
    print("="*80 + "\n")
    
    # Optimizer and scheduler
    print("[DEBUG] Loading perplexity metric...")
    metric = load('perplexity', experiment_id=str(uuid.uuid4()))
    print("[DEBUG] Metric loaded successfully")
    
    print("[DEBUG] Creating optimizer...")
    optimizer = AdamW(params=model.parameters(), lr=args.LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print("[DEBUG] Optimizer created successfully")
    
    print("[DEBUG] Creating learning rate scheduler...")
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(WARMUP_RATIO * total_training_steps), 
        num_training_steps=total_training_steps
    )
    print("[DEBUG] LR scheduler created successfully")
    
    # Setup beta_s2 annealing
    beta_s2_start = args.beta_s2_start if args.beta_s2_start is not None else args.BETA_S2
    beta_s2_end = args.beta_s2_end if args.beta_s2_end is not None else args.BETA_S2
    beta_s2_warmup_epochs = args.beta_s2_warmup_epochs
    
    if beta_s2_warmup_epochs > 0:
        print(f"[INFO] Beta_S2 annealing enabled: {beta_s2_start:.6f} -> {beta_s2_end:.6f} over {beta_s2_warmup_epochs} epochs")
    else:
        print(f"[INFO] Beta_S2 fixed at {args.BETA_S2}")
    
    beta_mse = float(args.BETA_S2_MSE)
    print(f"[DEBUG] Beta MSE set: beta_mse={beta_mse}")
    
    # Training loop
    train_losses = {'Task': [], 'Info': [], 'MSE': [], 'Total': []}
    test_performances = []
    best_total_loss = float('inf')
    best_model_state = None
    
    print("[DEBUG] About to start training loop...")
    print("Starting training for Stage 2")
    print(f"Total training steps: {total_training_steps}")
    print(f"Evaluation frequency: {args.EVAL_FREQ} epochs\n")
    
    print("[DEBUG] Entering training loop...")
    for epoch in range(args.EPOCHS):
        print(f"\n[DEBUG] ========== Starting Epoch {epoch+1}/{args.EPOCHS} ===========")
        
        # Compute annealed beta_s2 for current epoch
        if beta_s2_warmup_epochs > 0 and epoch < beta_s2_warmup_epochs:
            # Linear warmup from beta_s2_start to beta_s2_end
            progress = epoch / beta_s2_warmup_epochs
            beta = beta_s2_start + progress * (beta_s2_end - beta_s2_start)
        else:
            # After warmup, use final value
            beta = beta_s2_end
        
        print(f"[INFO] Epoch {epoch+1}: beta_s2 = {beta:.6f}")
        
        model.train()
        epoch_task_loss = 0
        epoch_info_loss = 0
        epoch_mse_loss = 0
        epoch_total_loss = 0
        
        print(f"[DEBUG] Epoch {epoch+1}: About to enumerate train_dataloader...")
        for step, batch in enumerate(train_dataloader):
            if step == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Step {step}: Got first batch from dataloader")
                print(f"[DEBUG]   Batch keys: {batch.keys()}")
                print(f"[DEBUG]   input_ids shape: {batch['input_ids'].shape if 'input_ids' in batch else 'N/A'}")
            
            # Move batch to device
            if step == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Step {step}: Moving batch to device...")
            batch = {k: v.to(device) for k, v in batch.items()}
            if step == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Step {step}: Batch moved to device")
            
            # Feature extraction from pre-trained language model (single forward pass)
            if step == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Step {step}: About to call base_model...")
            with torch.no_grad():
                outputs = base_model(
                    batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True
                )
            
            if step == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Step {step}: Base model forward pass completed")
            
            hidden_states = torch.stack(outputs.hidden_states)
            # Transform to batch-first and skip embedding layer
            hidden_states = hidden_states[1:].permute(1, 0, 2, 3)  # (batch, layers, seq, hidden)
            
            # Store last hidden state for MSE loss computation (reuse from this forward pass)
            llama3_last_hidden = outputs.hidden_states[-1].float()
            
            hidden_states = hidden_states.float()
            
            # Clear cache after base model forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get Stage 1 conditioning
            with torch.no_grad():
                _, mu1, _ = stage1_vib(
                    hidden_states if layer_s1 == "all" else hidden_states[:, layer_s1:layer_s1+1],
                    m=batch["attention_mask"], 
                    noise=False 
                )
            
            # Forward Stage 2 VIB model
            outputs_vib = model(
                hidden_states if layer_s2 == "all" else hidden_states[:, layer_s2:layer_s2+1],
                m=batch["attention_mask"],
                cond=mu1, 
                noise=not args.NO_IB
            )
            logits, merged_latents, mu, var = outputs_vib
            
            
            info_loss = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=-1)
            info_loss = torch.masked_select(info_loss, batch["attention_mask"].bool()).mean()
            
            # Compute MSE regression to LLaMA3's last hidden layer
            mse_loss = F.mse_loss(merged_latents, llama3_last_hidden, reduction='none').mean(dim=-1)
            mse_loss = torch.masked_select(mse_loss, batch["attention_mask"].bool()).mean()
            
            # Compute language modeling loss
            shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
            shift_labels = batch['labels'][:, 1:].contiguous()  # [batch, seq_len-1]
            
            # Flatten for loss calculation
            batch_size, seq_len, vocab_size = shift_logits.shape
            shift_logits = shift_logits.view(batch_size * seq_len, vocab_size)
            shift_labels = shift_labels.view(batch_size * seq_len)
            
            # Ignore padding tokens (pad_token_id is already mapped to -100 in collate_fn)
            task_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            # Total loss
            total_loss = task_loss + beta * info_loss + beta_mse * mse_loss
            
            # Check for NaN
            if torch.isnan(total_loss).any():
                print(f"ERROR: NaN detected in total_loss at epoch {epoch+1}, step {step}")
                print(f"  task_loss: {task_loss.item():.4f}")
                print(f"  info_loss: {info_loss.item() if not torch.isnan(info_loss) else 'NaN'}")
                print(f"  mse_loss: {mse_loss.item():.4f}")
                print(f"  beta: {beta}, beta_mse: {beta_mse}")
                print(f"  mu stats - min: {mu.min().item():.6e}, max: {mu.max().item():.6e}, mean: {mu.mean().item():.6e}")
                print(f"  var stats - min: {var.min().item():.6e}, max: {var.max().item():.6e}, mean: {var.mean().item():.6e}")
                print("  Stopping training to prevent NaN propagation.")
                break
            
            # Check for extreme loss values
            if total_loss.item() > 1e4:
                print(f"WARNING: Extremely large loss at epoch {epoch+1}, step {step}: {total_loss.item():.4f}")
                print(f"  task_loss: {task_loss.item():.4f}")
                print(f"  info_loss: {info_loss.item():.4f}")
                print(f"  mse_loss: {mse_loss.item():.4f}")
                print(f"  beta: {beta}, beta_mse: {beta_mse}")
                print(f"  Breakdown: task={task_loss.item():.4f} + beta*info={beta}*{info_loss.item():.4f} + beta_mse*mse={beta_mse}*{mse_loss.item():.4f}")
                print(f"  mu stats - min: {mu.min().item():.6e}, max: {mu.max().item():.6e}, mean: {mu.mean().item():.6e}")
                print(f"  var stats - min: {var.min().item():.6e}, max: {var.max().item():.6e}, mean: {var.mean().item():.6e}")
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Track losses
            epoch_task_loss += task_loss.item()
            epoch_info_loss += info_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_total_loss += total_loss.item()
        
        # Compute average losses
        avg_total_loss = epoch_total_loss / len(train_dataloader)
        avg_task_loss = epoch_task_loss / len(train_dataloader)
        avg_info_loss = epoch_info_loss / len(train_dataloader)
        avg_mse_loss = epoch_mse_loss / len(train_dataloader)
        
        train_losses['Task'].append(avg_task_loss)
        train_losses['Info'].append(avg_info_loss)
        train_losses['MSE'].append(avg_mse_loss)
        train_losses['Total'].append(avg_total_loss)
        
        # Save best model
        if not np.isnan(avg_total_loss) and avg_total_loss < best_total_loss:
            best_total_loss = avg_total_loss
            best_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1}/{args.EPOCHS}, Task Loss: {avg_task_loss:.4f}, "
                  f"Info Loss: {avg_info_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, "
                  f"Total Loss: {avg_total_loss:.4f} -> New best!")
        else:
            print(f"Epoch {epoch+1}/{args.EPOCHS}, Task Loss: {avg_task_loss:.4f}, "
                  f"Info Loss: {avg_info_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, "
                  f"Total Loss: {avg_total_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % args.EVAL_FREQ == 0:
            model.eval()
            all_predictions = []
            all_references = []
            
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
                
                # Get Stage 1 conditioning
                with torch.no_grad():
                    _, mu1, _ = stage1_vib(
                        hidden_states if layer_s1 == "all" else hidden_states[:, layer_s1:layer_s1+1],
                        m=batch["attention_mask"]
                    )
                    
                    # Forward Stage 2 VIB model
                    outputs_vib = model(
                        hidden_states if layer_s2 == "all" else hidden_states[:, layer_s2:layer_s2+1],
                        m=batch["attention_mask"], 
                        cond=mu1
                    )
                
                logits = outputs_vib[0]
                
                # Compute perplexity
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
            
            # Compute perplexity
            if all_predictions:
                avg_log_prob = np.mean(all_predictions)
                perplexity = np.exp(-avg_log_prob)
                print(f"  Test Perplexity: {perplexity:.4f}")
                test_performances.append(perplexity)
            else:
                print("  No valid predictions for perplexity calculation")
                test_performances.append(float('inf'))
            
            model.train()
    
    # Save results
    postfix = f"_bs={args.BATCH_SIZE}_lr={args.LEARNING_RATE}_dim={args.LATENT_DIM}"
    if args.NO_IB:
        postfix += "_noib" 
    else:
        postfix += f"_b={args.BETA_S1}_{args.BETA_S2}"
    postfix += f"_layer={args.LAYER_S1}_{args.LAYER_S2}"
    
    print(f"\nSaving results with postfix: {postfix}")
    
    with open(f"{SAVE_REPORTS_PATH}train_losses{postfix}.pkl", 'wb') as f:
        pickle.dump(train_losses, f)
    
    with open(f"{SAVE_REPORTS_PATH}test_perplexity{postfix}.pkl", 'wb') as f:
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
    
    # Output final alpha value from the decoder
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'alpha'):
        final_alpha = model.decoder.alpha.item()
        print(f"\nFinal alpha value (Stage 2 weight): {final_alpha:.6f}")
        print(f"Final Stage 1 weight: {1 - final_alpha:.6f}")
    
    print(f"\nStage 2 training completed!")
    print(f"Best model saved with Total Loss: {best_total_loss:.4f}")
    print(f"Final test perplexity: {test_performances[-1] if test_performances else 'N/A'}")


if __name__ == "__main__":
    main()
