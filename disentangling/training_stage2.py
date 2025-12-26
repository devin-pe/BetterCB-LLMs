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


def build_position_vocab(dataset):
    """Build vocabulary of unique position patterns"""
    unique_positions = set()
    for sample in dataset:
        unique_positions.add(sample['position'])
    position_to_idx = {pos: idx for idx, pos in enumerate(sorted(unique_positions))}
    return position_to_idx


def position_strings_to_indices(position_strings, position_to_idx):
    """Convert position strings to class indices"""
    indices = [position_to_idx[pos_str] for pos_str in position_strings]
    return torch.tensor(indices, dtype=torch.long)


def has_pii_in_position(position_string):
    """Check if position string contains any PII (any 1s)"""
    return any(int(x) == 1 for x in position_string.split(','))


def prepare_dataset_stage2_with_position(batch, tokenizer, max_length=512):
    """Prepare batch for Stage 2 but keep position field for adversarial classification"""
    tokenized = tokenizer(
        batch["text"],
        padding=False,
        truncation=True,
        max_length=max_length
    )
    
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'],
        'position': batch['position']  # Keep position for adversarial classification
    }


def create_collate_fn_with_position(tokenizer):
    """Collate function that preserves position field"""
    def collate_fn(batch):
        input_ids_list = [torch.tensor(x["input_ids"]) for x in batch]
        attention_mask_list = [torch.tensor(x["attention_mask"]) for x in batch]
        labels_list = [torch.tensor(x["labels"]) for x in batch]
        position_list = [x["position"] for x in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position": position_list
        }
    return collate_fn


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
    parser.add_argument("--GAMMA", type=float, default=2.5)
    parser.add_argument("--var_clamp_min", type=float, default=0.1,
                        help='Minimum variance clamp value used in the encoder (default: 0.1)')
    parser.add_argument("--var_clamp_max", type=float, default=10.0,
                        help='Maximum variance clamp value used in the encoder (default: 10.0)')
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--NO_IB", action='store_true')
    parser.add_argument("--MAX_LENGTH", type=int, default=512)
    parser.add_argument("--BATCH_SIZE", type=int, default=4)
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--EVAL_FREQ", type=int, default=10)
    parser.add_argument("--ADV_STEPS", type=int, default=3,
                        help='Number of adversarial classifier training steps per batch (default: 3)')
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
    SAVE_REPORTS_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/reports/vib/alternate_2/{DATA_}/{args.MODEL_NAME}/"
    SAVE_MODEL_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/models/vib/alternate_2/{DATA_}/{args.MODEL_NAME}/"
    
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
    
    # # Extract RMSNorm from LLaMA3 for normalizing merged_latents
    # # Using the final layer norm from the base model ensures proper scale
    # llama3_norm = base_model.norm  # LLaMA's final RMSNorm layer
    # llama3_norm.eval()  # Keep frozen during training
    print(f"Extracted RMSNorm from LLaMA3 base model (will be applied after merging)")
    
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
    
    # Load datasets
    print("Loading datasets...")
    # Load with stage="1" to get position labels (for token-level PII mask)
    train_data_raw = load_echr_data('train', stage="1", data_path=DATA_PATH)
    test_data_raw = load_echr_data('test', stage="1", data_path=DATA_PATH)
    
    # Adversarial classifier for token-level training
    # Predicts which sample (0 to batch_size-1) a PII token belongs to
    # This forces encoder to remove sample-specific PII patterns
    max_batch_size = args.BATCH_SIZE  # Maximum number of classes (samples in batch)
    adversarial_classifier = torch.nn.Sequential(
        torch.nn.Linear(LATENT_DIM, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(512, max_batch_size)  # Predict sample index within batch
    ).to(device)
    adversarial_classifier.train()
    print(f"Initialized token-level adversarial classifier (predicts sample index, max {max_batch_size} classes)")
    
    # Prepare datasets
    print("[DEBUG] About to map train_data...")
    train_data = train_data_raw.map(lambda batch: prepare_dataset_stage2_with_position(batch, tokenizer, args.MAX_LENGTH), batched=True)
    print("[DEBUG] Train data mapped successfully")
    
    print("[DEBUG] About to map test_data...")
    test_data = test_data_raw.map(lambda batch: prepare_dataset_stage2_with_position(batch, tokenizer, args.MAX_LENGTH), batched=True)
    print("[DEBUG] Test data mapped successfully")
    
    # Create data loaders
    print("[DEBUG] Creating collate_fn...")
    collate_fn = create_collate_fn_with_position(tokenizer)
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
    
    # Count PII tokens for monitoring
    print("[DEBUG] Counting PII tokens in training data...")
    total_pii_tokens = 0
    total_tokens = 0
    for sample in train_data:
        pos_labels = [int(x) for x in sample['position'].split(',')]
        total_pii_tokens += sum(pos_labels)
        total_tokens += len(pos_labels)
    pii_ratio = total_pii_tokens / total_tokens if total_tokens > 0 else 0
    print(f"[INFO] PII tokens: {total_pii_tokens} / {total_tokens} ({100*pii_ratio:.1f}%)")
    
    # Print run configuration
    print("\n" + "="*80)
    print("STAGE 2 TRAINING CONFIGURATION (Token-Level Adversarial Training)")

    print("="*80)
    print(f"Total Epochs: {args.EPOCHS}")
    print(f"Batch Size: {args.BATCH_SIZE}")
    print(f"Learning Rate: {args.LEARNING_RATE}")
    print(f"Stage 1 Latent Dimension: {stage1_latent_dim}")
    print(f"Stage 2 Latent Dimension: {LATENT_DIM}")
    print(f"Beta S2 (Adversarial): {args.BETA_S2}")
    print(f"Gamma (MSE weight): {args.GAMMA}")
    print(f"Layer S1: {args.LAYER_S1}, Layer S2: {args.LAYER_S2}")
    print(f"Adversarial Method: Alternating Training (classifier → encoder)")
    print(f"Adv steps per batch: {args.ADV_STEPS} classifier, 1 encoder")
    print(f"Model save location: {SAVE_MODEL_PATH}")
    print("="*80 + "\n")
    
    # Optimizer and scheduler
    print("[DEBUG] Loading perplexity metric...")
    metric = load('perplexity', experiment_id=str(uuid.uuid4()))
    print("[DEBUG] Metric loaded successfully")
    
    print("[DEBUG] Creating optimizers...")
    # Separate optimizers for alternating training
    vib_optimizer = AdamW(params=model.parameters(), lr=args.LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    adv_optimizer = AdamW(params=adversarial_classifier.parameters(), lr=args.LEARNING_RATE * 2, weight_decay=WEIGHT_DECAY)
    print("[DEBUG] Optimizers created successfully")
    
    print("[DEBUG] Creating learning rate scheduler...")
    vib_scheduler = get_cosine_schedule_with_warmup(
        vib_optimizer, 
        num_warmup_steps=int(WARMUP_RATIO * total_training_steps), 
        num_training_steps=total_training_steps
    )
    print("[DEBUG] LR scheduler created successfully")
    
    beta_s2 = float(args.BETA_S2)
    gamma = float(args.GAMMA)
    print(f"[INFO] Beta S2 (adversarial loss): {beta_s2}")
    print(f"[DEBUG] Gamma (MSE weight): gamma={gamma}")
    
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
        
        model.train()
        adversarial_classifier.train()
        epoch_task_loss = 0
        epoch_adv_loss = 0
        epoch_mse_loss = 0
        epoch_total_loss = 0
        
        print(f"[DEBUG] Epoch {epoch+1}: About to enumerate train_dataloader...")
        for step, batch in enumerate(train_dataloader):
            if step == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Step {step}: Got first batch from dataloader")
                
            # Move batch to device (except position which is a list of strings)
            if step == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Step {step}: Moving batch to device...")
            position_strings = batch.pop('position')  # Remove position before moving to device
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['position'] = position_strings  # Add back position as list
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
            
            # Store layer 31 (pre-RMSNorm) for MSE loss computation - matches decoder's output space
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
            
            # Extract token-level position labels (binary: 1=PII, 0=non-PII)
            # Convert position strings to token-level binary masks
            batch_size = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]
            pii_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
            
            for i, pos_str in enumerate(batch['position']):
                pos_labels = [int(x) for x in pos_str.split(',')]
                # Truncate or pad to match sequence length
                pos_labels = pos_labels[:seq_len] + [0] * (seq_len - len(pos_labels))
                pii_mask[i] = torch.tensor(pos_labels, dtype=torch.bool, device=device)
            
            # Forward Stage 2 VIB model
            outputs_vib = model(
                hidden_states if layer_s2 == "all" else hidden_states[:, layer_s2:layer_s2+1],
                m=batch["attention_mask"],
                cond=mu1, 
                noise=not args.NO_IB
            )
            logits, merged_latents, mu, var = outputs_vib
            
            # Apply RMSNorm to merged_latents before decoding (matches LLaMA3's output normalization)
            # with torch.no_grad():
            #     merged_latents_normalized = llama3_norm(merged_latents)
            
            # Token-level adversarial training:
            # Only train on tokens where position=1 (PII entities)
            # Combine PII mask with attention mask (only valid PII tokens)
            valid_pii_mask = pii_mask & batch["attention_mask"].bool()
            num_pii_tokens = valid_pii_mask.sum().item()
            
            if num_pii_tokens > 0:
                # Extract latents only at PII positions [num_pii_tokens, latent_dim]
                pii_latents = mu[valid_pii_mask]  # Shape: [num_pii_tokens, 4096]
                
                # For each PII token, we want to predict which document it came from
                # This forces the encoder to remove document-specific PII patterns
                # Create labels: which sample each PII token belongs to
                sample_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
                pii_sample_labels = sample_indices[valid_pii_mask]  # [num_pii_tokens]
                
                # === PHASE 1: Train adversarial classifier (args.ADV_STEPS steps) ===
                # Classifier learns to predict which sample each PII token came from
                for adv_step in range(args.ADV_STEPS):
                    adv_logits = adversarial_classifier(pii_latents.detach())  # [num_pii_tokens, batch_size]
                    # Predict which sample (0 to batch_size-1) this PII token belongs to
                    adv_loss_cls = F.cross_entropy(adv_logits[:, :batch_size], pii_sample_labels)
                    
                    adv_optimizer.zero_grad()
                    adv_loss_cls.backward()
                    adv_optimizer.step()
                
                # === PHASE 2: Train encoder to produce sample-invariant PII representations ===
                # Encoder learns to make PII tokens indistinguishable across samples
                adv_logits_enc = adversarial_classifier(pii_latents)  # gradients flow to encoder
                # Uniform target: equal probability for all samples
                uniform_target = torch.ones(num_pii_tokens, batch_size, device=device) / batch_size
                # KL divergence pushes predictions toward uniform
                adv_loss = F.kl_div(
                    F.log_softmax(adv_logits_enc[:, :batch_size], dim=-1), 
                    uniform_target, 
                    reduction='batchmean'
                )
            else:
                # No PII tokens in this batch
                adv_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        
            # Compute MSE regression to LLaMA3's layer 31 (pre-RMSNorm)
            # We compare pre-norm merged_latents to pre-norm layer 31 output
            mse_loss = F.mse_loss(merged_latents, llama3_last_hidden, reduction='none').mean(dim=-1)
            mse_loss = torch.masked_select(mse_loss, batch["attention_mask"].bool()).mean()
            
            # # Now recompute logits with normalized merged_latents
            # # Decoder expects RMSNorm-normalized representations
            # logits = model.decoder.lm_head(merged_latents_normalized)
            
            # Compute language modeling loss
            shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
            shift_labels = batch['labels'][:, 1:].contiguous()  # [batch, seq_len-1]
            
            # Flatten for loss calculation
            batch_size, seq_len, vocab_size = shift_logits.shape
            shift_logits = shift_logits.view(batch_size * seq_len, vocab_size)
            shift_labels = shift_labels.view(batch_size * seq_len)
            
            # Ignore padding tokens (pad_token_id is already mapped to -100 in collate_fn)
            task_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            
            # Total loss: task + adversarial (encourages PII independence) + MSE (representation quality)
            total_loss = task_loss + beta_s2 * adv_loss + gamma * mse_loss
            
            # Check for NaN
            if torch.isnan(total_loss).any():
                print(f"ERROR: NaN detected in total_loss at epoch {epoch+1}, step {step}")
                print(f"  task_loss: {task_loss.item():.4f}")
                print(f"  adv_loss (GRL): {adv_loss.item() if not torch.isnan(adv_loss) else 'NaN'}")
                print(f"  mse_loss: {mse_loss.item():.4f}")
                print(f"  beta_s2: {beta_s2}, gamma: {gamma}")
                print(f"  mu stats - min: {mu.min().item():.6e}, max: {mu.max().item():.6e}, mean: {mu.mean().item():.6e}")
                print(f"  var stats - min: {var.min().item():.6e}, max: {var.max().item():.6e}, mean: {var.mean().item():.6e}")
                print("  Stopping training to prevent NaN propagation.")
                break
            
            # Check for extreme loss values
            if total_loss.item() > 1e4:
                print(f"WARNING: Extremely large loss at epoch {epoch+1}, step {step}: {total_loss.item():.4f}")
                print(f"  task_loss: {task_loss.item():.4f}")
                print(f"  adv_loss: {adv_loss.item():.4f}")
                print(f"  mse_loss: {mse_loss.item():.4f}")
                print(f"  beta_s2: {beta_s2}, gamma: {gamma}")
                print(f"  Breakdown: task={task_loss.item():.4f} + beta_s2*adv={beta_s2}*{adv_loss.item():.4f} + gamma*mse={gamma}*{mse_loss.item():.4f}")
                print(f"  mu stats - min: {mu.min().item():.6e}, max: {mu.max().item():.6e}, mean: {mu.mean().item():.6e}")
                print(f"  var stats - min: {var.min().item():.6e}, max: {var.max().item():.6e}, mean: {var.mean().item():.6e}")
            
            # Backward pass with gradient clipping (for VIB only, adversarial already trained)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            vib_optimizer.step()
            vib_scheduler.step()
            vib_optimizer.zero_grad()
            
            # Track losses
            epoch_task_loss += task_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_total_loss += total_loss.item()
        
        # Compute average losses
        avg_total_loss = epoch_total_loss / len(train_dataloader)
        avg_task_loss = epoch_task_loss / len(train_dataloader)
        avg_adv_loss = epoch_adv_loss / len(train_dataloader)
        avg_mse_loss = epoch_mse_loss / len(train_dataloader)
        
        train_losses['Task'].append(avg_task_loss)
        train_losses['Info'].append(avg_adv_loss)  # Reuse 'Info' key for adversarial loss
        train_losses['MSE'].append(avg_mse_loss)
        train_losses['Total'].append(avg_total_loss)
        
        # Save best model
        if not np.isnan(avg_total_loss) and avg_total_loss < best_total_loss:
            best_total_loss = avg_total_loss
            best_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1}/{args.EPOCHS}, Task Loss: {avg_task_loss:.4f}, "
                  f"Adv Loss (Alt): {avg_adv_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, "
                  f"Total Loss: {avg_total_loss:.4f} -> New best!")
        else:
            print(f"Epoch {epoch+1}/{args.EPOCHS}, Task Loss: {avg_task_loss:.4f}, "
                  f"Adv Loss (Alt): {avg_adv_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, "
                  f"Total Loss: {avg_total_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % args.EVAL_FREQ == 0:
            model.eval()
            all_predictions = []
            all_references = []
            
            for batch in test_dataloader:
                # Remove position before moving to device (it's a list of strings)
                if 'position' in batch:
                    batch.pop('position')
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
                    
                    # # Apply RMSNorm to merged_latents before decoding (same as training)
                    # _, merged_latents, _, _ = outputs_vib
                    # merged_latents_normalized = llama3_norm(merged_latents)
                    # logits = model.decoder.lm_head(merged_latents_normalized)
                    
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
        alpha_tensor = model.decoder.alpha.detach().cpu()
        if alpha_tensor.numel() == 1:
            # Scalar alpha
            final_alpha = alpha_tensor.item()
            print(f"\nFinal alpha value (Stage 2 weight): {final_alpha:.6f}")
            print(f"Final Stage 1 weight: {1 - final_alpha:.6f}")
        else:
            # Multi-element alpha (e.g., per-token or per-layer)
            print(f"\nAlpha statistics (Stage 2 weight):")
            print(f"  Min: {alpha_tensor.min().item():.6f}")
            print(f"  Max: {alpha_tensor.max().item():.6f}")
            print(f"  Mean: {alpha_tensor.mean().item():.6f}")
            print(f"  Std: {alpha_tensor.std().item():.6f}")
            print(f"\nStage 1 weight statistics:")
            one_minus_alpha = 1.0 - alpha_tensor
            print(f"  Min: {one_minus_alpha.min().item():.6f}")
            print(f"  Max: {one_minus_alpha.max().item():.6f}")
            print(f"  Mean: {one_minus_alpha.mean().item():.6f}")
    
    print(f"\nStage 2 training completed!")
    print(f"Best model saved with Total Loss: {best_total_loss:.4f}")
    print(f"Final test perplexity: {test_performances[-1] if test_performances else 'N/A'}")


if __name__ == "__main__":
    main()
