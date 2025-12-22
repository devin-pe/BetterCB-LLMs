import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--STAGE", type=str, choices=["1", "2"])
parser.add_argument("--DATA_S1", type=str, default="custom_echr")
parser.add_argument("--DATA_S2", type=str, default="custom_echr")
parser.add_argument("--LATENT_DIM", type=int, default=128)
parser.add_argument("--MODEL_NAME", type=str, default="llama3")
parser.add_argument("--LAYER_S1", type=str, default="all")
parser.add_argument("--LAYER_S2", type=str, default="all")
parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
parser.add_argument("--BETA_S1", type=float, default=0.1)
parser.add_argument("--BETA_S2", type=float, default=0.1)
parser.add_argument("--BETA_S2_MSE", type=float, default=0.01)
parser.add_argument("--SEED", type=int, default=42)
parser.add_argument("--NO_IB", action='store_true')
parser.add_argument("--MAX_LENGTH", type=int, default=512)
args = parser.parse_args()

STAGE = args.STAGE
DATA_S1 = args.DATA_S1
DATA_S2 = args.DATA_S2
LATENT_DIM = args.LATENT_DIM
MODEL_NAME = args.MODEL_NAME
LAYER_S1 = args.LAYER_S1
LAYER_S2 = args.LAYER_S2
LEARNING_RATE = args.LEARNING_RATE
BETA_S1 = args.BETA_S1
BETA_S2 = args.BETA_S2
BETA_S2_MSE = args.BETA_S2_MSE
NO_IB = args.NO_IB
SEED = args.SEED
MAX_LENGTH = args.MAX_LENGTH

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(sys.modules[__name__].__file__), "..")))

DATA = DATA_S1 if STAGE == "1" else DATA_S2
print(f"Using dataset: {DATA}, Stage: {STAGE}")

# Task configuration
if STAGE == "1":
    OBJECTIVE = "position_labels"
else:
    OBJECTIVE = "next_word"
    
BATCH_SIZE = 4 
LAYER_S1 = LAYER_S1 if LAYER_S1 == "all" else int(LAYER_S1)
LAYER_S2 = LAYER_S2 if LAYER_S2 in ["all", None] else int(LAYER_S2)
BETA = BETA_S1 if STAGE == "1" else BETA_S2
EPOCHS = 10 if STAGE == "1" else 10
EVAL_FREQ = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
SELECTED_GPU = 0
DATA_ = DATA_S1 if STAGE == "1" else DATA_S1 + "_" + DATA_S2

print(f"Task objective: {OBJECTIVE}")

# Paths
DATA_PATH = "/home/dpereira/CB-LLMs/generation/dataset/"
LOAD_STAGE1_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/models/vib/4096_1/{DATA_S1}/{MODEL_NAME}/"
SAVE_REPORTS_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/reports/vib/0_{STAGE}/{DATA_}/{MODEL_NAME}/"
SAVE_MODEL_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/models/vib/0_{STAGE}/{DATA_}/{MODEL_NAME}/"
print(f"Model will be saved at {SAVE_MODEL_PATH}")
## Imports
import pickle
import numpy as np
import matplotlib.pyplot as plt
import uuid
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset, load_from_disk
from evaluate import load
from transformers import AutoTokenizer, LlamaModel, LlamaConfig
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
from modules import VIB, VIBConfig

if not os.path.exists(SAVE_REPORTS_PATH):
    os.makedirs(SAVE_REPORTS_PATH)
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)

## GPU
if torch.cuda.is_available():     
    device = torch.device(f"cuda:{SELECTED_GPU}")
    print('We will use the GPU:', torch.cuda.get_device_name(SELECTED_GPU))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

def load_echr_data(split):
    """Load ECHR dataset for the specified split"""
    data = pd.read_csv(f"{DATA_PATH}echr_{split}.csv")
    print(f"Loaded ECHR {split} dataset with {len(data)} samples")
    
    if STAGE == "1":
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

def prepare_dataset_stage1(batch):
    """Prepare batch for Stage 1 (PII classification or token-level detection)"""
    # Tokenize text
    tokenized = tokenizer(
        batch["text"], 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    batch['input_ids'] = tokenized['input_ids']
    batch['attention_mask'] = tokenized['attention_mask']
    

    position_labels = []
    for pos_str in batch['position']:
        labels = [int(x) for x in pos_str.split(',')]
        # Pad or truncate to MAX_LENGTH to match tokenized sequence
        if len(labels) < MAX_LENGTH:
            labels = labels + [0] * (MAX_LENGTH - len(labels))
        else:
            labels = labels[:MAX_LENGTH]
        position_labels.append(labels)
    batch['labels'] = position_labels

    return batch

def prepare_dataset_stage2(batch):
    """Prepare batch for Stage 2 (language modeling)"""
    # Tokenize text for language modeling - no padding here, do it in collate_fn
    tokenized = tokenizer(
        batch["text"], 
        padding=False,  # Don't pad here, will pad in collate_fn
        truncation=True, 
        max_length=MAX_LENGTH
    )
    
    # Return lists, not tensors - DataLoader will handle batching
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids']  # For language modeling
    }

def collate_fn(batch):
    """Collate function for DataLoader"""
    if STAGE == "1":
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

# Load trained stage 1 vib model if in stage 2
if STAGE == "2":
    postfix = f"_bs={BATCH_SIZE}_lr={LEARNING_RATE}_dim={LATENT_DIM}"
    if NO_IB:
        postfix += "_noib" 
    else:
        postfix += f"_b={BETA_S1}"
    postfix += f"_layer={LAYER_S1}"
    
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
    
    stage1_config = VIBConfig(
        input_dim=base_model.config.hidden_size,
        latent_dim=stage1_latent_dim,  
        stage="1",
        num_classes=2, 
        layer_weight_averaging=LAYER_S1 == "all",
        num_layers=base_model.config.num_hidden_layers if LAYER_S1 == "all" else None
    )
    stage1_vib = VIB(stage1_config)
    stage1_vib.load_state_dict(checkpoint)
    stage1_vib.to(device)
    stage1_vib.eval()
    print(f"Loaded Stage 1 VIB model successfully from {stage1_model_file}")

layer_weight_averaging = (STAGE == "1" and LAYER_S1 == "all") or (STAGE == "2" and LAYER_S2 == "all")
num_classes = 2 if STAGE == "1" else tokenizer.vocab_size 

if STAGE == "2":
    # Stage 2 latent_dim must match LLaMA3 hidden size for proper merging with Stage 1
    LATENT_DIM = base_model.config.hidden_size  # 4096 for LLaMA3
    print(f"Stage 1 latent_dim: {stage1_latent_dim}, Stage 2 latent_dim: {LATENT_DIM} (enforced to match LLaMA3 hidden size)")
    cond_dim = stage1_latent_dim
else:
    cond_dim = None

vib_config = VIBConfig(
    input_dim=base_model.config.hidden_size,
    latent_dim=LATENT_DIM,
    stage=STAGE,
    num_classes=num_classes,
    layer_weight_averaging=layer_weight_averaging,
    num_layers=base_model.config.num_hidden_layers if layer_weight_averaging else None,
    cond_dim=cond_dim  # Set conditioning dimension for Stage 2
)
model = VIB(vib_config)
model.to(device)
model.train()

# Load data
print("Loading datasets...")
train_data = load_echr_data('train')
test_data = load_echr_data('test')

# Prepare datasets
if STAGE == "1":
    train_data = train_data.map(prepare_dataset_stage1, batched=True)
    test_data = test_data.map(prepare_dataset_stage1, batched=True)
else:
    train_data = train_data.map(prepare_dataset_stage2, batched=True)
    test_data = test_data.map(prepare_dataset_stage2, batched=True)

print("Creating data loaders...")
# Create data loaders
train_dataloader = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    collate_fn=collate_fn, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=0  # Set to 0 to avoid multiprocessing pickling errors
) 
test_dataloader = DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    collate_fn=collate_fn, 
    shuffle=False, 
    pin_memory=True, 
    num_workers=0  # Set to 0 to avoid multiprocessing pickling errors
) 

print(f"Data loaders created. Computing training steps...")
training_steps = len(train_dataloader)
print(f"Training steps per epoch: {training_steps}")
total_training_steps = EPOCHS * training_steps

# Print run configuration
print("\n" + "="*80)
print("RUN CONFIGURATION")
print("="*80)
print(f"Total Epochs: {EPOCHS}")
print(f"Model save location: {SAVE_MODEL_PATH}")
print(f"Stage: {STAGE}")
print(f"Latent dimension: {LATENT_DIM}")
print(f"Beta (Stage 2): {BETA_S2}")
print(f"Beta MSE (Stage 2): {BETA_S2_MSE}")
print("="*80 + "\n")

# Load metrics & optimizer
if STAGE == "1":
    metric = load('accuracy', experiment_id=str(uuid.uuid4()))
else:
    metric = load('perplexity', experiment_id=str(uuid.uuid4()))

optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(WARMUP_RATIO * total_training_steps), 
    num_training_steps=total_training_steps
)

beta_reach_steps = (EPOCHS - 5) * training_steps
beta = 0.1 if BETA == "incremental" else float(BETA)
BETA_INCREMENT = (1.0 - beta) / beta_reach_steps if BETA == "incremental" else 0

print(f"Starting training for Stage {STAGE}")
print(f"Total training steps: {total_training_steps}")
print(f"Evaluation frequency: {EVAL_FREQ} epochs")

train_losses = {'Task': [], 'Info': [], 'Total': []}
test_performances = []
best_total_loss = float('inf')
best_model_state = None

for epoch in range(EPOCHS):
    model.train()
    epoch_task_loss = 0
    epoch_info_loss = 0
    epoch_mse_loss = 0
    epoch_total_loss = 0
    
    for step, batch in enumerate(train_dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        mse_loss = torch.tensor(0.0, device=device)
        
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
        
        # Store last hidden state for MSE loss computation in Stage 2 (reuse from this forward pass)
        llama3_last_hidden = outputs.hidden_states[-1].float() if STAGE == "2" else None
        
        hidden_states = hidden_states.float() # Convert to float32 for VIB model
        
        # Clear cache after base model forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Forward VIB model
        if STAGE == "1":
            # Stage 1: PII classification
            logits, mu, var = model(
                hidden_states if LAYER_S1 == "all" else hidden_states[:, LAYER_S1:LAYER_S1+1],
                m=batch["attention_mask"], 
                noise=not NO_IB
            )
        else:
            # Stage 2: Language modeling with conditioning from stage 1
            with torch.no_grad():
                _, mu1, var1 = stage1_vib(
                    hidden_states if LAYER_S1 == "all" else hidden_states[:, LAYER_S1:LAYER_S1+1],
                    m=batch["attention_mask"], 
                    noise=False 
                ) 
            outputs_vib = model(
                hidden_states if LAYER_S2 == "all" else hidden_states[:, LAYER_S2:LAYER_S2+1],
                m=batch["attention_mask"],
                cond=mu1, 
                noise=not NO_IB
            )
            logits, merged_latents, mu, var = outputs_vib
        if NO_IB:
            info_loss = torch.tensor(0.0, device=device)
        else:
            if STAGE == "2":
                # KL divergence in Stage 1 space: KL(q(z2)||q(z1))
                
                # kl_div = 0.5 * (var / var1 + (mu2_in_stage1 - mu1).pow(2) / var1 + torch.log(var1 / var) - 1)
                # info_loss = torch.masked_select(kl_div.sum(dim=-1), batch["attention_mask"].bool()).mean()
                info_loss = torch.tensor(0.0, device=device)

                # Compute MSE regression to LLaMA3's last hidden layer (reuse from initial forward pass)
                mse_loss = F.mse_loss(merged_latents, llama3_last_hidden, reduction='none').mean(dim=-1)
                mse_loss = torch.masked_select(mse_loss, batch["attention_mask"].bool()).mean()
                
                if torch.isnan(mse_loss).any():
                    print(f"WARNING: NaN detected in mse_loss at epoch {epoch+1}, step {step}")
                    print(f"  merged_latents: min={merged_latents.min():.4f}, max={merged_latents.max():.4f}, has_nan={torch.isnan(merged_latents).any()}")
                    print(f"  llama3_last_hidden: min={llama3_last_hidden.min():.4f}, max={llama3_last_hidden.max():.4f}, has_nan={torch.isnan(llama3_last_hidden).any()}")

            else:
                # Stage 1: KL divergence loss (per-token)
                info_loss = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=-1)
                # Apply mask for Stage 1 (sequence-level loss)
                info_loss = torch.masked_select(info_loss, batch["attention_mask"].bool()).mean()
        
        if STAGE == "1":
            # Token-level sequence labeling loss
            batch_size, seq_len, num_classes = logits.shape
            
            # Flatten for loss calculation
            flat_logits = logits.view(batch_size * seq_len, num_classes)  # [batch*seq, 2]
            flat_labels = batch['labels'].view(batch_size * seq_len)  # [batch*seq]
            
            # Create mask to ignore padding tokens
            flat_mask = batch["attention_mask"].view(batch_size * seq_len).bool()
            
            # Compute loss only on non-padded tokens
            task_loss = F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])
        else:
            # Language modeling loss 
            shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
            shift_labels = batch['labels'][:, 1:].contiguous()  # [batch, seq_len-1]
            
            # Flatten for loss calculation
            batch_size, seq_len, vocab_size = shift_logits.shape
            shift_logits = shift_logits.view(batch_size * seq_len, vocab_size)  # [batch*seq_len, vocab_size]
            shift_labels = shift_labels.view(batch_size * seq_len)  # [batch*seq_len]
            
            # Ignore padding tokens
            task_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=tokenizer.pad_token_id)
        
        if torch.isnan(task_loss).any():
            print(f"WARNING: NaN detected in task_loss at epoch {epoch+1}, step {step}")
            print(f"  shift_logits: min={shift_logits.min():.4f}, max={shift_logits.max():.4f}, has_nan={torch.isnan(shift_logits).any()}")
            print(f"  shift_labels: min={shift_labels.min()}, max={shift_labels.max()}")
        
        beta2 = BETA_S2_MSE if STAGE == "2" else 0.0
        total_loss = task_loss + beta * info_loss + beta2 * mse_loss
        
        if torch.isnan(total_loss).any():
            print(f"ERROR: NaN detected in total_loss at epoch {epoch+1}, step {step}")
            print(f"  task_loss: {task_loss.item():.4f}")
            print(f"  info_loss: {info_loss.item():.4f} (beta={beta:.4f})")
            if STAGE == "2":
                print(f"  mse_loss: {mse_loss.item():.4f} (beta2={beta2:.4f})")
            print(f"  total_loss: {total_loss.item():.4f}")
            print("  Stopping training to prevent NaN propagation.")
            break
        
        # Check for extreme loss values that could cause gradient explosion
        if total_loss.item() > 1e4:
            print(f"WARNING: Extremely large loss detected at epoch {epoch+1}, step {step}: {total_loss.item():.4f}")
            print(f"  task_loss: {task_loss.item():.4f}")
            print(f"  info_loss: {info_loss.item():.4f}")
            if STAGE == "2":
                print(f"  mse_loss: {mse_loss.item():.4f}")
            print("  Skipping this batch to prevent gradient explosion.")
            continue
        
        total_loss.backward()
        optimizer.step()
        
        lr_scheduler.step()
        optimizer.zero_grad()

        task_loss_val = task_loss.item()
        info_loss_val = info_loss.item() if not NO_IB else 0.0
        mse_loss_val = mse_loss.item() if STAGE == "2" else 0.0
        total_loss_val = total_loss.item()
        epoch_task_loss += task_loss_val
        if not NO_IB:
            epoch_info_loss += info_loss_val
        if STAGE == "2":
            epoch_mse_loss += mse_loss_val
        epoch_total_loss += total_loss_val
        
        if BETA == "incremental":
            beta = min(beta + BETA_INCREMENT, 1.0)

    avg_total_loss = epoch_total_loss / len(train_dataloader)
    avg_task_loss = epoch_task_loss / len(train_dataloader)
    avg_info_loss = epoch_info_loss / len(train_dataloader) if not NO_IB else 0.0
    avg_mse_loss = epoch_mse_loss / len(train_dataloader) if STAGE == "2" else 0.0
    
    train_losses['Task'].append(avg_task_loss)
    if not NO_IB:
        train_losses['Info'].append(avg_info_loss)
        train_losses['Total'].append(avg_total_loss)
    
    if np.isnan(avg_total_loss):
        print(f"ERROR: NaN detected in epoch {epoch+1} averages:")
        print(f"  avg_task_loss: {avg_task_loss:.4f}")
        print(f"  avg_info_loss: {avg_info_loss:.4f}")
        if STAGE == "2":
            print(f"  avg_mse_loss: {avg_mse_loss:.4f}")
        print(f"  avg_total_loss: {avg_total_loss:.4f}")
        print(f"  Best model NOT updated (keeping best from epoch with loss {best_total_loss:.4f})")
    
    if not np.isnan(avg_total_loss) and avg_total_loss < best_total_loss:
        best_total_loss = avg_total_loss
        best_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        print(f"  -> New best model saved (Total Loss: {avg_total_loss:.4f})")

    if STAGE == "2":
        print(f"Epoch {epoch+1}/{EPOCHS}, Task Loss: {epoch_task_loss/len(train_dataloader):.4f}, "
              f"Info Loss: {epoch_info_loss/len(train_dataloader):.4f}, "
              f"MSE Loss: {epoch_mse_loss/len(train_dataloader):.4f}, "
              f"Total Loss: {epoch_total_loss/len(train_dataloader):.4f}")
    else:
        print(f"Epoch {epoch+1}/{EPOCHS}, Task Loss: {epoch_task_loss/len(train_dataloader):.4f}, "
              f"Info Loss: {epoch_info_loss/len(train_dataloader):.4f}, "
              f"Total Loss: {epoch_total_loss/len(train_dataloader):.4f}")

    # Evaluation
    if (epoch + 1) % EVAL_FREQ == 0:
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
            # Convert to float32 for VIB model
            hidden_states = hidden_states.float()
            
            # Forward VIB model
            if STAGE == "1":
                with torch.no_grad():
                    logits, mu, var = model(
                        hidden_states if LAYER_S1 == "all" else hidden_states[:, LAYER_S1:LAYER_S1+1],
                        m=batch["attention_mask"]
                    )
            else:
                with torch.no_grad():
                    _, mu1, var1 = stage1_vib(
                        hidden_states if LAYER_S1 == "all" else hidden_states[:, LAYER_S1:LAYER_S1+1],
                        m=batch["attention_mask"]
                    ) 
                    outputs_vib = model(
                        hidden_states if LAYER_S2 == "all" else hidden_states[:, LAYER_S2:LAYER_S2+1],
                        m=batch["attention_mask"], 
                        cond=mu1, 
                    )
                logits = outputs_vib[0]  # Extract logits (outputs_vib is (logits, hidden_repr, mu, var))
                
            # Compute predictions
            if STAGE == "1":
                preds = torch.argmax(logits, dim=-1)  # [batch, seq_len] or [batch]
                batch_size, seq_len = preds.shape
                flat_preds = preds.view(batch_size * seq_len)
                flat_labels = batch['labels'].view(batch_size * seq_len)
                flat_mask = batch["attention_mask"].view(batch_size * seq_len).bool()
                
                # Only evaluate on non-padded tokens
                predictions = flat_preds[flat_mask].cpu().numpy()
                references = flat_labels[flat_mask].cpu().numpy()
 
            else:
                # For language modeling, compute perplexity
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['labels'][..., 1:].contiguous()
                
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Filter out padding tokens
                mask = shift_labels != tokenizer.pad_token_id
                if mask.sum() > 0:
                    filtered_logits = shift_logits[mask]
                    filtered_labels = shift_labels[mask]
                    
                    log_probs = F.log_softmax(filtered_logits, dim=-1)
                    token_log_probs = log_probs.gather(1, filtered_labels.unsqueeze(1)).squeeze(1)
                    
                    predictions = token_log_probs.cpu().numpy().tolist()
                    references = [1] * len(predictions)  # Dummy references for perplexity
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)
            
            if STAGE == "1":
                metric.add_batch(predictions=predictions, references=references)

        # Compute metrics
        if STAGE == "1":
            perf = metric.compute()['accuracy']
            print(f"Test Accuracy: {perf:.4f}")
        else:
            if all_predictions:
                avg_log_prob = np.mean(all_predictions)
                perplexity = np.exp(-avg_log_prob)
                perf = perplexity
                print(f"Test Perplexity: {perf:.4f}")
            else:
                perf = float('inf')
                print("No valid predictions for perplexity calculation")
        
        test_performances.append(perf)

# Save results first (before final evaluation to ensure model is saved even if evaluation crashes)
postfix = f"_bs={BATCH_SIZE}_lr={LEARNING_RATE}_dim={LATENT_DIM}"
if NO_IB:
    postfix += "_noib" 
else:
    postfix += f"_b={BETA_S1}" if STAGE == "1" else f"_b={BETA_S1}_{BETA_S2}"
postfix += f"_layer={LAYER_S1}" if STAGE == "1" else f"_layer={LAYER_S1}_{LAYER_S2}"

print(f"\nSaving results with postfix: {postfix}")

with open(f"{SAVE_REPORTS_PATH}train_losses{postfix}.pkl", 'wb') as f:
    pickle.dump(train_losses, f)

metric_name = "accuracy" if STAGE == "1" else "perplexity"
with open(f"{SAVE_REPORTS_PATH}test_{metric_name}{postfix}.pkl", 'wb') as f:
    pickle.dump(test_performances, f)

# Save best model (or final model if no best was tracked)
model_state_to_save = best_model_state if best_model_state is not None else model.state_dict()
torch.save(model_state_to_save, f'{SAVE_MODEL_PATH}model{postfix}.pth')
print(f"Saved best model with Total Loss: {best_total_loss:.4f}")

if layer_weight_averaging:
    # Load best model to get correct layer weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    layer_weights = torch.nn.functional.softmax(model.layer_weights, dim=0).detach().cpu().numpy().tolist()
    with open(f"{SAVE_MODEL_PATH}layer-weights{postfix}.pkl", 'wb') as f:
        pickle.dump(layer_weights, f)

# Load best model for final evaluation (if we have one that's better than current)
if best_model_state is not None:
    print(f"\nLoading best model (Total Loss: {best_total_loss:.4f}) for final evaluation...")
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Create a new test loader without multiprocessing to avoid pickling errors
    final_test_dataloader = DataLoader(
        test_data, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=0  # No multiprocessing for final evaluation
    )
    
    # Re-evaluate best model on test set
    if STAGE == "1":
        metric = load('accuracy', experiment_id=str(uuid.uuid4()))
    else:
        all_predictions = []
        all_references = []
    
    with torch.no_grad():
        for batch in final_test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = base_model(
                    batch["input_ids"], 
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True
                )
            
            hidden_states = torch.stack(outputs.hidden_states)
            hidden_states = hidden_states[1:].permute(1, 0, 2, 3)
            hidden_states = hidden_states.float()
            
            if STAGE == "1":
                with torch.no_grad():
                    logits, mu, var = model(
                        hidden_states if LAYER_S1 == "all" else hidden_states[:, LAYER_S1:LAYER_S1+1],
                        m=batch["attention_mask"]
                    )
            else:
                with torch.no_grad():
                    _, mu1, var1 = stage1_vib(
                        hidden_states if LAYER_S1 == "all" else hidden_states[:, LAYER_S1:LAYER_S1+1],
                        m=batch["attention_mask"]
                    ) 
                    outputs_vib = model(
                        hidden_states if LAYER_S2 == "all" else hidden_states[:, LAYER_S2:LAYER_S2+1],
                        m=batch["attention_mask"], 
                        cond=mu1, 
                    )
                logits = outputs_vib[0]
            
            if STAGE == "1":
                preds = torch.argmax(logits, dim=-1)
                batch_size, seq_len = preds.shape
                flat_preds = preds.view(batch_size * seq_len)
                flat_labels = batch['labels'].view(batch_size * seq_len)
                flat_mask = batch["attention_mask"].view(batch_size * seq_len).bool()
                
                predictions = flat_preds[flat_mask].cpu().numpy()
                references = flat_labels[flat_mask].cpu().numpy()
                metric.add_batch(predictions=predictions, references=references)
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['labels'][..., 1:].contiguous()
                
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                mask = shift_labels != tokenizer.pad_token_id
                if mask.sum() > 0:
                    filtered_logits = shift_logits[mask]
                    filtered_labels = shift_labels[mask]
                    
                    log_probs = F.log_softmax(filtered_logits, dim=-1)
                    token_log_probs = log_probs.gather(1, filtered_labels.unsqueeze(1)).squeeze(1)
                    
                    predictions = token_log_probs.cpu().numpy().tolist()
                    references = [1] * len(predictions)
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)
    
    # Compute best model performance
    if STAGE == "1":
        best_perf = metric.compute()['accuracy']
        print(f"Best Model Test Accuracy: {best_perf:.4f}")
        # Replace the last test_performances entry with best model performance
        if test_performances:
            test_performances[-1] = best_perf
    else:
        if all_predictions:
            avg_log_prob = np.mean(all_predictions)
            best_perplexity = np.exp(-avg_log_prob)
            print(f"Best Model Test Perplexity: {best_perplexity:.4f}")
            # Replace the last test_performances entry with best model performance
            if test_performances:
                test_performances[-1] = best_perplexity
        else:
            print("No valid predictions for perplexity calculation")
    
    # Update test_performances with final evaluation result
    if test_performances:
        if STAGE == "1":
            test_performances[-1] = best_perf
        else:
            if all_predictions:
                test_performances[-1] = best_perplexity
    
    # Re-save test performances with updated final evaluation
    metric_name = "accuracy" if STAGE == "1" else "perplexity"
    with open(f"{SAVE_REPORTS_PATH}test_{metric_name}{postfix}.pkl", 'wb') as f:
        pickle.dump(test_performances, f)

print(f"\nTraining completed for Stage {STAGE}")
print(f"Best model saved with Total Loss: {best_total_loss:.4f}")
print(f"Final performance: {test_performances[-1] if test_performances else 'N/A'}")