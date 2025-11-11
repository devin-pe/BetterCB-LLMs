import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from modules import CBL
from utils import eos_pooling
import evaluate
import time

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_length", type=int, default=350)
parser.add_argument("--num_workers", type=int, default=0)


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text):
        self.encoded_text = encoded_text


    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in self.encoded_text.items()}
        return t

    def __len__(self):
        return len(self.encoded_text['input_ids'])


def build_loaders(encoded_text, mode):
    dataset = ClassificationDataset(encoded_text)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=True if mode == "train" else False)
    return dataloader



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    print("loading data...")
    if args.dataset == 'custom_echr':
        print("Loading custom ECHR test dataset from CSV...")
        test_path = "/home/dpereira/CB-LLMs/generation/dataset/echr_test.csv"
        if os.path.exists(test_path):
            df = pd.read_csv(test_path)
            if 'fact' in df.columns and 'has_person' in df.columns:
                test_dataset = Dataset.from_dict({
                    'text': df['fact'].tolist(),
                    'label': df['has_person'].tolist()
                })
                print(f"Loaded ECHR test dataset with {len(test_dataset)} samples")
            else:
                raise ValueError(f"ECHR test CSV at {test_path} missing expected columns 'fact' and 'has_person'. Found: {list(df.columns)}")
        else:
            raise FileNotFoundError(f"ECHR test CSV not found at {test_path}")
    else:
        test_dataset = load_dataset(args.dataset, split='test')
        print("test data len: ", len(test_dataset))
    print("tokenizing...")
    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == 'ag_news':
        def replace_bad_string(example):
            example["text"] = example["text"].replace("#36;", "")
            example["text"] = example["text"].replace("#39;", "'")
            return example
        test_dataset = test_dataset.map(replace_bad_string)

    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[CFG.example_name[args.dataset]], padding=True, truncation=True,
                            max_length=args.max_length), batched=True, batch_size=len(test_dataset))
    encoded_test_dataset = encoded_test_dataset.remove_columns([CFG.example_name[args.dataset]])
    if args.dataset == 'SetFit/sst2':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if args.dataset == 'dbpedia_14':
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])
    encoded_test_dataset = encoded_test_dataset[:len(encoded_test_dataset)]

    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("creating loader...")
    test_loader = build_loaders(encoded_test_dataset, mode="test")

    print("preparing backbone")
    
    # Load the full CBLLM model like eval_perplexity.py does
    if args.dataset == 'custom_echr':
        # model_path = "models/from_pretained_llama3_lora_cbm/custom_echr"
        model_path = "/scratch-shared/tmp.ISacU0WbVs/custom_echr"
        print(f"Loading CBLLM model from: {model_path}")
        
        # Load the base model and CBL components
        config = LlamaConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Loading base model...")
        preLM = LlamaModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"  # Let it handle device placement automatically
        )
        print("Base model loaded successfully")
        preLM.eval()
        
        unsup_dim = CFG.unsup_dim.get(args.dataset, CFG.unsup_dim.get('default', config.hidden_size))
        print(f"Creating CBL with unsup_dim={unsup_dim}")
        cbl = CBL(config, len(concept_set), tokenizer, unsup_dim=unsup_dim)
        
        # Load CBL weights
        cbl_path = os.path.join(model_path, "cbl_epoch_8.pt")
        
        print(f"Loading CBL weights from: {cbl_path}")
        cbl.load_state_dict(torch.load(cbl_path, map_location='cpu'))
        cbl.eval()
        
        # Move CBL to same device as base model
        model_device = next(preLM.parameters()).device if hasattr(preLM, 'parameters') else device
        cbl.to(model_device)
        print(f"CBL moved to device: {model_device}")
        
    else:
        # Logic for other datasets  
        epoch_num = CFG.epoch[args.dataset]
        peft_path = f"from_pretained_llama3_lora_cbm/{args.dataset.replace('/', '_')}/llama3_epoch_{epoch_num}"
        cbl_path = f"from_pretained_llama3_lora_cbm/{args.dataset.replace('/', '_')}/cbl_epoch_{epoch_num}.pt"
        preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
        preLM.load_adapter(peft_path)
        preLM.eval()
        cbl = CBL(config, len(concept_set), tokenizer).to(device)
        cbl.load_state_dict(torch.load(cbl_path, map_location=device))
        cbl.eval()

    print("eval concepts...")
    metric = evaluate.load("accuracy")
    concept_predictions = []
    total_batches = len(test_loader)
    print(f"Processing {total_batches} batches...")
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx % 10 == 0:  # Print progress every 10 batches
            print(f"Processing batch {batch_idx + 1}/{total_batches}")
        
        # Move batch to appropriate device (let the model handle it with device_map)
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            # Move features to CBL device
            model_device = next(cbl.parameters()).device
            features = features.to(model_device)
            concepts, _, _ = cbl(features.float())
        concept_predictions.append(eos_pooling(concepts, batch["attention_mask"].to(model_device)))
        
        # Clean up memory periodically for large unsupervised layer
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    print("Concatenating predictions...")
    concept_predictions = torch.cat(concept_predictions, dim=0).detach().cpu()
    pred = np.argmax(concept_predictions.numpy(), axis=-1)
    metric.add_batch(predictions=pred, references=encoded_test_dataset["label"])
    print(metric.compute())
