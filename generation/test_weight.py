import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import config as CFG
from datasets import load_dataset, concatenate_datasets
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from modules import CBL


parser = argparse.ArgumentParser()

device = torch.device("cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    tokenizer.pad_token = tokenizer.eos_token

    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("preparing backbone")
    peft_path = "from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') + "/llama3"
    cbl_path = "from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') + "/cbl.pt"
    preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    preLM.load_adapter(peft_path)
    preLM.eval()
    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device))
    cbl.eval()

    w = cbl.fc.weight.data[:, :len(concept_set)].T
    for i in range(len(concept_set)):
        top_values, top_ids = torch.topk(w[i], k=10)
        print("Neuron: ", concept_set[i])
        print("Top 10 tokens with highest weight:")
        for j in range(10):
            print("Neuron:", concept_set[i], "[",round(float(top_values.detach().cpu()[j]), 3), "]", tokenizer.decode(top_ids[j]))

    print((w > 1e-6).count_nonzero() / w.numel())

