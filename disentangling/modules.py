import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
generation_dir = os.path.join(project_root, "generation")

if generation_dir not in sys.path:
    sys.path.insert(0, generation_dir)
    
import torch
from torch import nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model, GPT2TokenizerFast, RobertaModel
import torch.nn.functional as F
from utils import top_k_top_p_filtering
from dataclasses import dataclass
from typing import Optional



class SelfAttention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = torch.nn.Parameter(torch.ones(1, embed_dim), requires_grad=True)

    def forward(self, x, attention_mask=None, output_attentions=False):
        """Input shape: Batch x Time x Hidden Dim"""
       
        scores = torch.matmul(x, self.query.unsqueeze(0).transpose(-2, -1)).squeeze(-1)  # (batch_size, seq_len)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == False, float('-inf')) 

        attn_weights = torch.nn.functional.softmax(scores, dim=-1).unsqueeze(-1)  
        
        # weighted average
        pooled_x = torch.sum(attn_weights * x, dim=1) 
        
        outputs = (pooled_x, attn_weights) if output_attentions else (pooled_x,)
        return outputs
    

@dataclass
class VIBConfig(): 
    input_dim: Optional[int] = None
    latent_dim: Optional[int] = None
    num_classes: Optional[int] = None
    stage: Optional[str] = None
    layer_weight_averaging: Optional[bool] = False
    num_layers: Optional[int] = None
    cond_dim: Optional[int] = None  # Dimension of Stage 1 conditioning (for Stage 2 only)

class VariationalEncoder(torch.nn.Module):
    def __init__(self, config):
        super(VariationalEncoder, self).__init__()
        self.enc1 = torch.nn.Linear(config.input_dim, config.input_dim)
        self.enc2 = torch.nn.Linear(config.input_dim, config.input_dim)
        self.mu = torch.nn.Linear(config.input_dim, config.latent_dim)
        self.var = torch.nn.Linear(config.input_dim, config.latent_dim)

    def forward(self, h):
        o = F.gelu(self.enc1(h))
        o = F.gelu(self.enc2(o))
        
        mu = self.mu(o)
        var = F.softplus(self.var(o)) # to generate positive values
        
        return mu, var

class CBLDecoder(torch.nn.Module):
    def __init__(self, config):
        super(CBLDecoder, self).__init__()
        self.lm_head = torch.nn.Linear(config.latent_dim, config.num_classes)
    
    def forward(self, z, m=None, cond=None):
        # z: [batch, seq_len, latent_dim]
        # Per-token prediction: Apply classifier to each token position
        # For sequence labeling tasks (e.g., PERSON token detection)
        logits = self.lm_head(z)  # [batch, seq_len, num_classes]
        outputs = (logits,)
        return outputs
    
class TextDecoder(torch.nn.Module):
    def __init__(self, config):
        super(TextDecoder, self).__init__()
        self.latent_dim = config.latent_dim
        
        # Projection layer for Stage 1 conditioning if dimensions don't match
        self.cond_dim = getattr(config, 'cond_dim', config.latent_dim)
        if self.cond_dim != self.latent_dim:
            self.cond_projection = torch.nn.Linear(self.cond_dim, self.latent_dim)
        else:
            self.cond_projection = None
        
        self.clf = torch.nn.Linear(config.latent_dim*2, config.num_classes) # latent_dim * 2 due to concatenation
    
    def forward(self, z, m, cond):
        # z: [batch, seq_len, latent_dim] - Stage 2 latent representation
        # cond: [batch, seq_len, cond_dim] - Stage 1 latent representation (mu from encoder)
        
        # Project conditioning to match Stage 2 latent dimension if needed
        if self.cond_projection is not None:
            cond = self.cond_projection(cond)  # [batch, seq_len, cond_dim] -> [batch, seq_len, latent_dim]
        
        batch_size, seq_len, latent_dim = z.shape
        
        concatenated = torch.cat([cond, z], dim=-1)  # [batch, seq_len, latent_dim*2]
        
        logits = self.clf(concatenated)  # [batch, seq_len, num_classes]
        
        outputs = (logits,)
        return outputs

class VIB(torch.nn.Module):
    def __init__(self, config):
        super(VIB, self).__init__()
        self.layer_weight_averaging = config.layer_weight_averaging
        if self.layer_weight_averaging:
            self.layer_weights = torch.nn.Parameter(torch.ones(config.num_layers)/config.num_layers, requires_grad=True)

        self.encoder = VariationalEncoder(config)

        if config.stage == "1":
            self.decoder = CBLDecoder(config)
        elif config.stage == "2":
            self.decoder = TextDecoder(config)
        else:
            raise ValueError("Invalid VIB training stage!")

    def forward(self, h, m=None, cond=None, noise=True): 
        if self.layer_weight_averaging:
            # compute weighted sum over layers
            w = torch.nn.functional.softmax(self.layer_weights, dim=0)
            h = torch.sum(h * w.view(1, w.shape[0], 1, 1), dim=1)

        mu, var = self.encoder(h)
        std = var ** 0.5
        # reparameterization trick: introducing epsilon only during training, and use the z = mu during inference
        if self.training and noise:
            eps = torch.randn_like(std) # sample from N(0, 1)
            z = mu + std * eps 
        else:
            z = mu

        outputs = self.decoder(z, m, cond)
        
        return outputs + (mu, var)
    
    def generate(self, ids, preLM, cond=None, mask=None, intervene=None, length=100, temp=0.7, topk=100, topp=0.9, repetition_penalty=1.5, eos_token_id=128001):
        """
        Generate text autoregressively using the VIB model.
        
        Args:
            ids: Initial token ids [batch_size, initial_seq_len]
            preLM: Pre-trained language model (e.g., LlamaModel)
            cond: Optional conditioning from Stage 1 VIB [batch_size, latent_dim]
            mask: Optional attention mask
            intervene: Not used (kept for API compatibility)
            length: Maximum number of tokens to generate
            temp: Temperature for sampling
            topk: Top-k filtering parameter
            topp: Top-p (nucleus) filtering parameter
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: Token ID to stop generation
            
        Returns:
            generated_ids: Full sequence including initial ids [batch_size, total_length]
            final_vib_outputs: VIB outputs from last generation step
        """
        past_key_values = None
        generated_ids = ids.clone()
        
        for i in range(length):
            # Forward pass through base LM with caching
            outputs = preLM(
                generated_ids[:, -1:] if past_key_values is not None else generated_ids, 
                past_key_values=past_key_values, 
                use_cache=True, 
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values
            
            # Get all hidden states and transform to correct shape
            hidden_states = torch.stack(outputs.hidden_states)  # [num_layers, batch_size, seq_len, hidden_size]
            # Skip embedding layer and permute to [batch_size, num_layers, seq_len, hidden_size]
            hidden_states = hidden_states[1:].permute(1, 0, 2, 3)
            
            # Pass through VIB model with conditioning
            outputs_vib = self.forward(hidden_states, m=mask, cond=cond)
            logits = outputs_vib[0]  # [batch_size, seq_len, vocab_size]
            
            # Get logits for the last position
            next_token_logits = logits[:, -1, :].clone()  # [batch_size, vocab_size]
            
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    # If score < 0, multiply by penalty; if score > 0, divide by penalty
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            next_token_logits = next_token_logits / temp
            
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            
            # Sample next token
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat((generated_ids, next_token), dim=-1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return generated_ids, outputs_vib