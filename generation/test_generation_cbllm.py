import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
import config as CFG
from transformers import LlamaConfig, LlamaModel, AutoTokenizer
from modules import CBL
from utils import eos_pooling
import evaluate
import time

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dataset", type=str, default="SetFit/sst2")
parser.add_argument("--max_length", type=int, default=1024)
parser.add_argument("--gen_length", type=int, default=100, help="Number of tokens to generate with CBL.generate")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parser.parse_args()

    # config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')


    concept_set = CFG.concepts_from_labels[args.dataset]
    print("concept len: ", len(concept_set))

    print("preparing backbone")
    peft_path = "models/from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') # + "/llama3"
    cbl_path = "models/from_pretained_llama3_lora_cbm/" + args.dataset.replace('/', '_') +  "/cbl_epoch_2.pt"
    # preLM = LlamaModel.from_pretrained('meta-llama/Meta-Llama-3-8B', torch_dtype=torch.bfloat16).to(device)
    # preLM.load_adapter(peft_path)
    config = LlamaConfig.from_pretrained(peft_path)
    tokenizer = AutoTokenizer.from_pretrained(peft_path)
    tokenizer.pad_token = tokenizer.eos_token
    preLM = LlamaModel.from_pretrained(peft_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    preLM.eval()
    cbl = CBL(config, len(concept_set), tokenizer).to(device)
    cbl.load_state_dict(torch.load(cbl_path, map_location=device))
    cbl.eval()

    input_ids = torch.tensor([tokenizer.encode("")]).to(device)

    print("generation...")
    #v = [0] * len(concept_set)
    # For SST2 and Yelp, intervene v[0] to a large value will give you negative movie or Yelp review
    # For AGnews, there are four classes World news, Sport news, Business news and Tech news, intervene v[0], ..., v[3] accordingly will give you the sentences related to the concepts
    # v[0] = 100
    with torch.no_grad():
        text_ids, concept_activation = cbl.generate(input_ids, preLM, length=args.gen_length) #intervene=v, length=args.gen_length)
    print(tokenizer.decode(text_ids[0]))
    # print(concept_activation)
