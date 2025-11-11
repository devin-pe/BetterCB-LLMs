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
parser.add_argument("--dataset", type=str, default="yelp_polarity")
parser.add_argument("--max_length", type=int, default=1024)

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

    # text = "Contrary to other reviews, I have zero complaints about the service or the prices. I have been getting tire service here for the past 5 years now, and compared to my experience with places like Pep Boys, these guys are experienced and know what they're doing."
    # text = "The food is good. Unfortunately the service is very hit or miss. The main issue seems to be with the kitchen, the waiters and waitresses are often very apologetic for the long waits and it's pretty obvious that some of them avoid the tables after taking the initial order to avoid hearing complaints."
    text = "I went there today! The cut was terrible! I have an awful experience. They lady that cut my hair was nice but she wanted to leave early so she made a disaster in my head!"
    input_ids = torch.tensor([tokenizer.encode(text)]).to(device)
    length = input_ids.shape[1]
    activation = []
    with torch.no_grad():
        for i in range(length):
            features = preLM(input_ids=input_ids[:, :i+1]).last_hidden_state
            concepts, _, _ = cbl(features.float())
            activation.append(F.softmax(concepts[0, -1, :], dim=-1))
    activation = torch.stack(activation, dim=0)
    encoded = tokenizer(text)
    token_word_map = []
    for word_id in encoded.word_ids():
        if word_id is not None:
            start, end = encoded.word_to_tokens(word_id)
            tokens = [start, end - 1]
            if len(token_word_map) == 0 or token_word_map[-1] != tokens:
                token_word_map.append(tokens)
    text = []
    token_list = []
    for w in input_ids[0]:
        token_list.append(tokenizer.decode(w))
    for m in token_word_map:
        t = "".join(token_list[m[0]:m[1] + 1])
        text.append(t)
    print(text)
    for i in range(len(concept_set)):
        a = activation[:, i]
        probs = []
        for m in token_word_map:
            probs.append(round(float(torch.mean(a[m[0]:m[1]+1])), 2))
        print(concept_set[i]+": ", probs)