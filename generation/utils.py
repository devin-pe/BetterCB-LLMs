import torch
import torch.nn.functional as F
import config as CFG

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def eos_pooling(token_embeddings, attention_mask):
    last_index = []
    for i in range(attention_mask.size(0)):
        last_index.append(check_zero(attention_mask[i]))
    last_index = torch.tensor(last_index)
    return token_embeddings[range(len(last_index)), last_index]

def check_zero(mask):
    for i in range(len(mask)):
        if mask[i] == 0:
            return i-1
    return len(mask)-1

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=float('-inf')):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0][indices_to_remove] = filter_value
    return logits

def elastic_net_penalty(param, alpha=0.99):
    return alpha * torch.abs(param).mean() + (1-alpha) * torch.square(param).mean()