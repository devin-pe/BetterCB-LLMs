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
    
BATCH_SIZE = 4 #if STAGE == "1" else 2
LAYER_S1 = LAYER_S1 if LAYER_S1 == "all" else int(LAYER_S1)
LAYER_S2 = LAYER_S2 if LAYER_S2 in ["all", None] else int(LAYER_S2)
BETA = BETA_S1 if STAGE == "1" else BETA_S2
EPOCHS = 10 if STAGE == "1" else 15
EVAL_FREQ = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.005
MAX_GRAD_NORM = 1
SELECTED_GPU = 0
DATA_ = DATA_S1 if STAGE == "1" else DATA_S1 + "_" + DATA_S2

print(f"Task objective: {OBJECTIVE}")

# Paths
DATA_PATH = "/home/dpereira/CB-LLMs/generation/dataset/"
LOAD_STAGE1_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/models/vib/1/{DATA_S1}/{MODEL_NAME}/"
SAVE_REPORTS_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/reports/vib/{STAGE}/{DATA_}/{MODEL_NAME}/"
SAVE_MODEL_PATH = f"{os.environ['HOME']}/CB-LLMs/disentangling/models/vib/{STAGE}/{DATA_}/{MODEL_NAME}/"

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
        # For language modeling, we'll use the text as both input and target (shifted)
        dataset = Dataset.from_dict({
            'text': texts,
            'labels': texts 
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
    # Tokenize text for language modeling
    tokenized = tokenizer(
        batch["text"], 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    batch['input_ids'] = tokenized['input_ids']
    batch['attention_mask'] = tokenized['attention_mask']
    
    # For language modeling, labels are the same as input_ids but shifted
    # Handle the shifting in the loss calculation
    batch['labels'] = tokenized['input_ids'].clone()
    
    return batch

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
        input_ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        labels = torch.stack([torch.tensor(x["labels"]) for x in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

model_path = "/home/dpereira/CB-LLMs/analysing_pii_leakage/examples/experiments/experiment_00015"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = LlamaModel.from_pretrained(model_path)
base_model.to(device)
base_model.eval()

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
    print(f"Stage 1 latent_dim: {stage1_latent_dim}, Stage 2 latent_dim: {LATENT_DIM}")
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
test_data = load_echr_data('validation')  # Using validation as test

# Prepare datasets
if STAGE == "1":
    train_data = train_data.map(prepare_dataset_stage1, batched=True)
    test_data = test_data.map(prepare_dataset_stage1, batched=True)
else:
    train_data = train_data.map(prepare_dataset_stage2, batched=True)
    test_data = test_data.map(prepare_dataset_stage2, batched=True)

# Create data loaders
train_dataloader = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    collate_fn=collate_fn, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=2
) 
test_dataloader = DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    collate_fn=collate_fn, 
    shuffle=False, 
    pin_memory=True, 
    num_workers=2
) 

training_steps = len(train_dataloader)
total_training_steps = EPOCHS * training_steps

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

# Training loop
train_losses = {'Task': [], 'Info': [], 'Total': []}
test_performances = []

for epoch in range(EPOCHS):
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
                    noise=False  # No noise for conditioning
                ) 
            outputs_vib = model(
                hidden_states if LAYER_S2 == "all" else hidden_states[:, LAYER_S2:LAYER_S2+1],
                m=batch["attention_mask"],
                cond=mu1, 
                noise=not NO_IB
            )
            logits, mu, var = outputs_vib        # Info loss (KL divergence)
        if NO_IB:
            info_loss = torch.tensor(0.0, device=device)
        else:
            if STAGE == "2":
                # Stage 2: Minimize I(Z1, Z2) via KL(q(z2|h) || q(z1|h))
                # Project mu1 and var1 to Stage 2 latent dimension if needed
                if model.decoder.cond_projection is not None:
                    mu1_proj = model.decoder.cond_projection(mu1)
                    var1_proj = model.decoder.cond_projection(var1)
                else:
                    mu1_proj = mu1
                    var1_proj = var1
                
                # KL(N(mu2, var2) || N(mu1_proj, var1_proj))
                info_loss = 0.5 * torch.sum(
                    torch.log(var1_proj / var) + (var + (mu - mu1_proj).pow(2)) / var1_proj - 1,
                    dim=-1
                )
            else:
                # Stage 1: Minimize I(Z1, H) via KL(q(z1|h) || N(0,1))
                info_loss = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var, dim=-1)
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
        
        # Total loss
        total_loss = task_loss + beta * info_loss 

        # Optimization
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Record losses
        epoch_task_loss += task_loss.item()
        if not NO_IB:
            epoch_info_loss += info_loss.item()
        epoch_total_loss += total_loss.item()
        
        if BETA == "incremental":
            beta = min(beta + BETA_INCREMENT, 1.0)

    # Store epoch averages
    train_losses['Task'].append(epoch_task_loss / len(train_dataloader))
    if not NO_IB:
        train_losses['Info'].append(epoch_info_loss / len(train_dataloader))
        train_losses['Total'].append(epoch_total_loss / len(train_dataloader))

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
                logits, _, _ = outputs_vib  # Unpack (logits, mu, var)
                
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

# Save results
postfix = f"_bs={BATCH_SIZE}_lr={LEARNING_RATE}_dim={LATENT_DIM}"
if NO_IB:
    postfix += "_noib" 
else:
    postfix += f"_b={BETA_S1}" if STAGE == "1" else f"_b={BETA_S1}_{BETA_S2}"
postfix += f"_layer={LAYER_S1}" if STAGE == "1" else f"_layer={LAYER_S1}_{LAYER_S2}"

print(f"Saving results with postfix: {postfix}")

with open(f"{SAVE_REPORTS_PATH}train_losses{postfix}.pkl", 'wb') as f:
    pickle.dump(train_losses, f)

metric_name = "accuracy" if STAGE == "1" else "perplexity"
with open(f"{SAVE_REPORTS_PATH}test_{metric_name}{postfix}.pkl", 'wb') as f:
    pickle.dump(test_performances, f)

# Save model
torch.save(model.state_dict(), f'{SAVE_MODEL_PATH}model{postfix}.pth')

if layer_weight_averaging:
    layer_weights = torch.nn.functional.softmax(model.layer_weights, dim=0).detach().cpu().numpy().tolist()
    with open(f"{SAVE_MODEL_PATH}layer-weights{postfix}.pkl", 'wb') as f:
        pickle.dump(layer_weights, f)

print(f"Training completed for Stage {STAGE}")
print(f"Final performance: {test_performances[-1] if test_performances else 'N/A'}")