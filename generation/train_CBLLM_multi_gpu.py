import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from modules import CBL
import time
from utils import elastic_net_penalty, mean_pooling
import pandas as pd
import random
import gc

# Memory optimization utilities
def cleanup_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Set memory optimizations
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()

# Multi-GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_multi_gpu = torch.cuda.device_count() > 1
print(f"Available GPUs: {torch.cuda.device_count()}")
if use_multi_gpu:
    print(f"Using multi-GPU training with {torch.cuda.device_count()} GPUs")
else:
    print(f"Using single GPU: {device}")

parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=1)  # Reduced for memory efficiency
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text):
        self.encoded_text = encoded_text


    def __getitem__(self, idx):
        # Get the item from the HuggingFace dataset
        item = self.encoded_text[idx]
        # Convert to tensors
        t = {key: torch.tensor(value) for key, value in item.items()}
        return t

    def __len__(self):
        return len(self.encoded_text)

def build_loaders(dataset, mode):
    dataset = ClassificationDataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              shuffle=True, pin_memory=False, drop_last=True)  # Disable pin_memory for memory efficiency
    return dataloader

args = parser.parse_args()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

# Load tokenizer and config from the fine-tuned model for consistency
fine_tuned_model_path = "/home/dpereira/CB-LLMs/generation/analysing_pii_leakage/examples/experiments/experiment_00015"
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

config = LlamaConfig.from_pretrained(fine_tuned_model_path)

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)

if args.dataset == 'SetFit/sst2':
    print("loading SetFit/sst2 dataset...")
    data = load_dataset(args.dataset)
    train_dataset = data["train"]
    val_dataset = data["validation"]

elif args.dataset == 'custom_echr':
    print("loading data...")
    print("Loading custom ECHR dataset with has_person labels...")
    
    # Load the separate train and validation datasets
    train_data = pd.read_csv("/home/dpereira/CB-LLMs/generation/dataset/echr_train.csv")
    val_data = pd.read_csv("/home/dpereira/CB-LLMs/generation/dataset/echr_validation.csv")
    
    print(f"Loaded ECHR training dataset with {len(train_data)} samples")
    print(f"Loaded ECHR validation dataset with {len(val_data)} samples")
    
    # Print distribution of has_person labels
    print(f"Training has_person distribution: {train_data['has_person'].value_counts().to_dict()}")
    print(f"Validation has_person distribution: {val_data['has_person'].value_counts().to_dict()}")
    
    train_dataset = Dataset.from_dict({
        'text': train_data['fact'].tolist(),
        'label': train_data['has_person'].tolist()
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_data['fact'].tolist(),
        'label': val_data['has_person'].tolist()
    })
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Print label distribution for training set
    train_labels = np.array(train_dataset['label'])
    print(f"Training label distribution: {np.bincount(train_labels)}")
    
    # Print a sample text
    print(f"Sample text: {train_dataset[0]['text'][:200]}...")

print("training data len: ", len(train_dataset))

print("tokenizing...")
if args.dataset in ['SetFit/sst2', 'custom_echr']:
    encoded_train_dataset = train_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True,
        batch_size=len(train_dataset))
    encoded_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    if args.dataset in ['SetFit/sst2', 'custom_echr']:
        encoded_val_dataset = val_dataset.map(
            lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True, max_length=args.max_length), batched=True,
            batch_size=len(val_dataset))
        encoded_val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        encoded_val_dataset = encoded_val_dataset[:len(encoded_val_dataset)]

concept_set = CFG.concepts_from_labels[args.dataset]
print("concept len: ", len(concept_set))

print("creating loader...")
train_loader = build_loaders(encoded_train_dataset, mode="train")
if args.dataset in ['SetFit/sst2', 'custom_echr']:
    val_loader = build_loaders(encoded_val_dataset, mode="valid")

print("preparing backbone")
# Use the fine-tuned Llama3 model from experiment_00015
print(f"Loading fine-tuned model from: {fine_tuned_model_path}")

# Load model with automatic device mapping (let it use full GPU memory)
if use_multi_gpu:
    preLM = LlamaModel.from_pretrained(
        fine_tuned_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Let accelerate handle the distribution
        low_cpu_mem_usage=True
    )
    print(f"Model loaded with automatic device mapping across {torch.cuda.device_count()} GPUs")
else:
    preLM = LlamaModel.from_pretrained(
        fine_tuned_model_path, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)
print("Fine-tuned model loaded successfully")

# Enable gradient checkpointing with error handling
try:
    preLM.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")
except Exception as e:
    print(f"Warning: Could not enable gradient checkpointing: {e}")

# Clean memory after model loading
cleanup_memory()
print("Memory cleaned after model loading")

# Memory-efficient optimizer with reduced state and lower learning rate
opt_prelm = torch.optim.AdamW(
    preLM.parameters(), 
    lr=1e-5,  # Reduced from 5e-5
    weight_decay=0.01,
    eps=1e-6,  # Smaller epsilon for memory efficiency
    foreach=False  # Disable foreach for memory efficiency
)
print("Optimizer configured for all model parameters")

# Place CBL and classifier on specific GPUs based on available count
gpu_count = torch.cuda.device_count()
print(f"Configuring for {gpu_count} GPUs")

if use_multi_gpu and gpu_count >= 3:
    # For 3+ GPUs: Put CBL on GPU 0 and classifier on GPU 2 (balanced)
    cbl_device = torch.device("cuda:0")
    classifier_device = torch.device("cuda:2") 
    print("Placing CBL on GPU 0 and classifier on GPU 2 for 3-GPU setup")
elif use_multi_gpu and gpu_count >= 2:
    # For 2 GPUs: Put CBL on GPU 1 and classifier on GPU 0
    cbl_device = torch.device("cuda:1")
    classifier_device = torch.device("cuda:0") 
    print("Placing CBL on GPU 1 and classifier on GPU 0 for 2-GPU setup")
else:
    # Single GPU fallback
    cbl_device = device
    classifier_device = device
    print("Using single GPU setup")

# Force CUDA synchronization to ensure proper initialization
torch.cuda.synchronize()
print("CUDA synchronized")

unsup_dim = CFG.unsup_dim.get(args.dataset, CFG.unsup_dim.get('default', config.hidden_size))
print(f"Creating CBL with unsup_dim={unsup_dim} on device {cbl_device}")

try:
    cbl = CBL(config, len(concept_set), tokenizer, unsup_dim=unsup_dim).to(cbl_device)
    print("CBL created and moved to device successfully")
    
    # Print memory usage after CBL creation
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
    
    opt_cbl = torch.optim.AdamW(
        cbl.parameters(), 
        lr=1e-5,  # Reduced from 5e-5
        weight_decay=0.01,
        eps=1e-6,
        foreach=False
    )
    print("CBL optimizer created successfully")
    
except Exception as e:
    print(f"Error creating CBL: {e}")
    raise

print("preparing classifier")
try:
    classifier = torch.nn.Linear(unsup_dim, len(concept_set)).to(classifier_device)
    print(f"Classifier created on device {classifier_device}")
    
    opt_classifier = torch.optim.AdamW(
        classifier.parameters(), 
        lr=1e-4,  # Reduced from 1e-3
        weight_decay=0.01,
        eps=1e-6,
        foreach=False
    )
    print("Classifier optimizer created successfully")
    
    # Print final memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            cached = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i} final: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
    
except Exception as e:
    print(f"Error creating classifier: {e}")
    raise

print("start training...")
best_loss = float('inf')
d_name = args.dataset.replace('/', '_')

for epoch in range(CFG.epoch[args.dataset]):
    print("Epoch ", epoch+1, ":")
    preLM.train()
    cbl.train()
    classifier.train()
    if args.dataset in ['SetFit/sst2', 'custom_echr']:
        training_concept_loss = []
        training_word_loss = []
        training_neg_entropy_loss = []
        training_reg_loss = []
        
        # Initialize for gradient accumulation
        accumulated_steps = 0
        
        for i, batch in enumerate(train_loader):
            try:
                if i == 0:
                    print(f"Processing first batch (batch size: {batch['input_ids'].shape[0]})")
                
                # Move batch to appropriate device - let the model handle device placement
                batch = {k: v.to(device) for k, v in batch.items()}
                concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
                word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
                
                if i == 0:
                    print("Running model forward pass...")
                
                # Forward
                features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                
                if i == 0:
                    print(f"Features shape: {features.shape}, device: {features.device}")
                
                # Move features to CBL device if using multi-GPU
                if use_multi_gpu and torch.cuda.device_count() >= 2:
                    features_for_cbl = features.to(cbl_device)
                    attention_mask_for_classifier = batch["attention_mask"].to(classifier_device)
                else:
                    features_for_cbl = features
                    attention_mask_for_classifier = batch["attention_mask"]
                
                if i == 0:
                    print(f"Running CBL forward pass on device {cbl_device}...")
                    
                concepts, unsup, vocabs = cbl(features_for_cbl.float())
                
                if i == 0:
                    print(f"CBL output shapes - concepts: {concepts.shape}, unsup: {unsup.shape}, vocabs: {vocabs.shape}")
                
                # Move labels to the appropriate devices for loss calculation
                if use_multi_gpu and torch.cuda.device_count() >= 2:
                    concept_label_device = concept_label.to(cbl_device)
                    word_label_device = word_label.to(cbl_device)  # vocabs comes from CBL
                else:
                    concept_label_device = concept_label
                    word_label_device = word_label
                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                cleanup_memory()
                continue
            

            fc_weight = cbl.fc.weight
            
            concept_loss = torch.nn.CrossEntropyLoss()(concepts[:, :-1, :].reshape(-1, len(concept_set)), concept_label_device.reshape(-1))
            word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label_device.reshape(-1))
            loss = concept_loss + word_loss
            reg = elastic_net_penalty(fc_weight[:, :len(concept_set)])
            loss += 1.0 * reg
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            
            # Backward
            loss.backward()
            accumulated_steps += 1
            
            # Update weights every gradient_accumulation_steps
            if accumulated_steps % args.gradient_accumulation_steps == 0:
                opt_prelm.step()
                torch.cuda.empty_cache()
                opt_cbl.step()
                torch.cuda.empty_cache()
                opt_prelm.zero_grad()
                opt_cbl.zero_grad()
                torch.cuda.empty_cache()

            classification = classifier(mean_pooling(unsup.detach().to(classifier_device), attention_mask_for_classifier))
            discrimination_loss = torch.nn.CrossEntropyLoss()(classification, batch["label"].to(classifier_device))
            discrimination_loss = discrimination_loss / args.gradient_accumulation_steps
            discrimination_loss.backward(inputs=list(classifier.parameters()))
            
            if accumulated_steps % args.gradient_accumulation_steps == 0:
                opt_classifier.step()
                opt_classifier.zero_grad()
                torch.cuda.empty_cache()

            _, unsup, _ = cbl(features_for_cbl.detach().float())
            classification = classifier(mean_pooling(unsup.to(classifier_device), attention_mask_for_classifier))
            p = F.softmax(classification, dim=-1)
            neg_entropy_loss = torch.sum(p * torch.log(p), dim=-1).mean()
            neg_entropy_loss = neg_entropy_loss / args.gradient_accumulation_steps
            
            # Handle multi-GPU case for unsup parameters
            unsup_params = cbl.unsup.parameters()
            neg_entropy_loss.backward(inputs=list(unsup_params))
            
            if accumulated_steps % args.gradient_accumulation_steps == 0:
                opt_cbl.step()
                opt_cbl.zero_grad()
                
                # Clean memory periodically
                if i % 20 == 0:
                    cleanup_memory()

            print("batch", str(i), "concept loss:", (concept_loss * args.gradient_accumulation_steps).detach().cpu().numpy(), "word loss:", (word_loss * args.gradient_accumulation_steps).detach().cpu().numpy(), "neg e loss:", (neg_entropy_loss * args.gradient_accumulation_steps).detach().cpu().numpy(), "reg loss:", reg.detach().cpu().numpy(), end="\r")
            
            concept_loss_val = (concept_loss * args.gradient_accumulation_steps).detach().cpu().numpy()
            word_loss_val = (word_loss * args.gradient_accumulation_steps).detach().cpu().numpy()
            neg_entropy_loss_val = (neg_entropy_loss * args.gradient_accumulation_steps).detach().cpu().numpy()
            
            # Check for NaN/Inf and stop training if detected
            if (torch.isnan(concept_loss).any() or torch.isnan(word_loss).any() or torch.isnan(neg_entropy_loss).any() or
                torch.isinf(concept_loss).any() or torch.isinf(word_loss).any() or torch.isinf(neg_entropy_loss).any()):
                print(f"\nNaN/Inf detected at batch {i}! Stopping training to prevent waste.")
                print(f"concept_loss: {concept_loss_val}, word_loss: {word_loss_val}, neg_entropy_loss: {neg_entropy_loss_val}")
                break
            
            training_concept_loss.append(concept_loss_val)
            training_word_loss.append(word_loss_val)
            training_neg_entropy_loss.append(neg_entropy_loss_val)
            training_reg_loss.append(reg.detach().cpu().numpy())
            
            torch.cuda.empty_cache()
        
        # Final update if needed
        if accumulated_steps % args.gradient_accumulation_steps != 0:
            opt_prelm.step()
            opt_cbl.step()
            opt_classifier.step()
            opt_prelm.zero_grad()
            opt_cbl.zero_grad()
            opt_classifier.zero_grad()
        avg_training_concept_loss = sum(training_concept_loss)/len(training_concept_loss)
        avg_training_word_loss = sum(training_word_loss) / len(training_word_loss)
        avg_training_neg_entropy_loss = sum(training_neg_entropy_loss) / len(training_neg_entropy_loss)
        avg_training_reg_loss = sum(training_reg_loss)/len(training_reg_loss)
        print("training concept loss:", avg_training_concept_loss, "training word loss:", avg_training_word_loss, "training neg e loss:", avg_training_neg_entropy_loss, "training reg loss: ", avg_training_reg_loss)


        if args.dataset == 'SetFit/sst2':
            preLM.eval()
            cbl.eval()
            classifier.eval()
            validation_concept_loss = []
            validation_word_loss = []
            validation_neg_entropy_loss = []
            validation_reg_loss = []
            for i, batch in enumerate(val_loader):
                with torch.no_grad():
                    batch = {k: v.to(device) for k, v in batch.items()}
                    concept_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["label"].view(-1, 1))
                    word_label = torch.where(batch["attention_mask"][:, :-1] == 0, -100, batch["input_ids"][:, 1:])
                    features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                    
                    # Move features to CBL device if using multi-GPU
                    if use_multi_gpu and torch.cuda.device_count() >= 2:
                        features_for_cbl = features.to(cbl_device)
                    else:
                        features_for_cbl = features
                        
                    concepts, unsup, vocabs = cbl(features_for_cbl.float())
                    
                    # Move labels to the appropriate devices for loss calculation
                    if use_multi_gpu and torch.cuda.device_count() >= 2:
                        concept_label_device = concept_label.to(cbl_device)
                        word_label_device = word_label.to(cbl_device)
                    else:
                        concept_label_device = concept_label
                        word_label_device = word_label
                    
                    # Handle multi-GPU case
                    fc_weight = cbl.fc.weight
                    
                    concept_loss = torch.nn.CrossEntropyLoss()(concepts[:, :-1, :].reshape(-1, len(concept_set)), concept_label_device.reshape(-1))
                    
                    word_loss = torch.nn.CrossEntropyLoss()(vocabs[:, :-1, :].reshape(-1, config.vocab_size), word_label_device.reshape(-1))
                    loss = concept_loss + word_loss
                    reg = elastic_net_penalty(fc_weight[:, :len(concept_set)])
                    loss += 1.0 * reg

                    validation_concept_loss.append(concept_loss.detach().cpu().numpy())
                    validation_word_loss.append(word_loss.detach().cpu().numpy())
                    validation_reg_loss.append(reg.detach().cpu().numpy())
            avg_val_concept_loss = sum(validation_concept_loss) / len(validation_concept_loss)
            avg_val_word_loss = sum(validation_word_loss) / len(validation_word_loss)
            avg_val_reg_loss = sum(validation_reg_loss) / len(validation_reg_loss)
            print("validation concept loss:", avg_val_concept_loss, "validation word loss:", avg_val_word_loss, "validation reg loss: ", avg_val_reg_loss)

    # Check for NaN values and handle them
    if not (torch.isnan(torch.tensor(avg_training_concept_loss)) or torch.isinf(torch.tensor(avg_training_concept_loss))):
        if avg_training_concept_loss < best_loss:
            best_loss = avg_training_concept_loss
            # checkpoint_dir = f"/scratch-shared/tmp.ISacU0WbVs/{d_name}"
            checkpoint_dir = f"./models/4096/{d_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Critical saves that must succeed - fail fast if any fail
            try:
                preLM.save_pretrained(checkpoint_dir)
                print(f"Saved preLM HF-format at: {checkpoint_dir}")
            except Exception as e:
                raise RuntimeError(f"CRITICAL ERROR: Failed to save preLM to {checkpoint_dir}: {e}")

            try:
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"Saved tokenizer at: {checkpoint_dir}")
            except Exception as e:
                raise RuntimeError(f"CRITICAL ERROR: Failed to save tokenizer to {checkpoint_dir}: {e}")

            try:
                cbl_path = os.path.join(checkpoint_dir, f"cbl_epoch_{epoch + 1}.pt")
                torch.save(cbl.state_dict(), cbl_path)
                print(f"Saved CBL state to: {cbl_path}")
            except Exception as e:
                raise RuntimeError(f"CRITICAL ERROR: Failed to save CBL state to {checkpoint_dir}: {e}")

            print(f"All model components successfully saved for epoch {epoch} with loss {best_loss:.6f}")
    else:
        print(f"Warning: Training loss is NaN/Inf ({avg_training_concept_loss}), skipping model save")