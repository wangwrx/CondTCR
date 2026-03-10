import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from collections import Counter, defaultdict
from BERT import BERT, DualTaskBERT, set_seed
from datetime import datetime

from GPT import GPT_Model, get_attn_pad_mask, get_attn_subsequence_mask
from Data_prepare import make_data_for_seq2seq, make_data_for_gra
import argparse
from tqdm import tqdm
import logging
import math
import random
import Levenshtein
from scipy.stats import entropy
from transformers import (
    BeamSearchScorer, 
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    MinLengthLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor
)
"""
Current script supports four generation modes:
1. tkns                   - Regular tkns generation (one sequence at a time)
2. conditional_beam       - Classic beam search
3. conditional_beam + HF  - Use --use_hf_beam to enable HuggingFace BeamSearchScorer
4. acs                    - Adaptive Contrastive Search

Only code required for the above modes is kept, other modes have been removed.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

GENERATION_CONFIG = {
    'normal_threshold': 200,
    'fewshot_threshold': 50,
    
    'sampling_params': {
        'top_p': 0.92,
        'top_k': 60,
        'repetition_penalty': 1.2
    }
}

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism"""
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.3)
    def forward(self,Q,K,V,attn_mask):
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(64)
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.drop(nn.Softmax(dim=-1)(scores))
        context=torch.matmul(attn,V)
        return context,attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12, d_k=64, d_v=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_K=nn.Linear(d_model,d_k*n_heads,bias=False)
        self.W_V=nn.Linear(d_model,d_v*n_heads,bias=False)
        self.fc=nn.Linear(n_heads*d_v,d_model,bias=False)
        self.norm=nn.LayerNorm(self.d_model)
    def forward(self,input_Q,input_K,input_V,attn_mask):
        batch_size=input_Q.size(0)
        residual=input_Q
        Q=self.W_Q(input_Q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        K=self.W_K(input_K).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V=self.W_V(input_V).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2)
        attn_mask=attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context=context.transpose(1,2).reshape(batch_size,-1,self.n_heads*self.d_v)
        output=self.fc(context)
        return self.norm(output + residual), attn
    
class PoswiseFeedForwardNet(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model=768):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model*4, bias=False),
            nn.GELU(),
            nn.Linear(d_model*4, d_model, bias=False)
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(0.3)
    def forward(self,inputs):
        residual = inputs
        output = self.drop(self.fc(inputs))
        return self.norm(output + residual)
    
class EncoderLayer(nn.Module):
    """Transformer encoder layer"""
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs,attn

class DecoderLayer(nn.Module):
    """Transformer decoder layer"""
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self,dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    """Transformer encoder stack"""
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    def forward(self,enc_inputs, bert_enc_outputs):
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            bert_enc_outputs, enc_self_attn = layer(bert_enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return bert_enc_outputs, enc_self_attns
    
class Decoder(nn.Module):
    """Transformer decoder stack"""
    def __init__(self, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    def forward(self,dec_inputs, enc_inputs, enc_outputs, gpt_dec_outputs):
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            gpt_dec_outputs, dec_self_attn, dec_enc_attn = layer(gpt_dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return gpt_dec_outputs, dec_self_attns, dec_enc_attns

class GRA_Enhanced(nn.Module):
    """Enhanced GRA model with null condition vector support"""
    def __init__(self, bert, gpt, vocabsize, d_model=768):
        super().__init__()
        self.bert = bert
        self.gpt = gpt
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocabsize, bias=False)
        
        self.null_condition = nn.Parameter(torch.randn(d_model))
        
    def forward(self, enc_inputs, dec_inputs, masked_pos):
        enc_outputs_bert, _ = self.bert(enc_inputs, masked_pos)
        
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_outputs_bert)
        
        dec_outputs_gpt, _ = self.gpt(dec_inputs)
        
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs, dec_outputs_gpt
        )
        
        dec_outputs = dec_outputs + dec_outputs_gpt
        
        dec_logits = self.projection(dec_outputs)
        
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

def categorize_test_samples_by_training_set(test_data_path, train_data_path):
    """
    Stratify test samples based on training set
    
    Args:
        test_data_path: Test set path
        train_data_path: Training set path
        
    Returns:
        test_categories: Category list for each test sample
        category_stats: Category statistics dictionary
    """
    test_df = pd.read_csv(test_data_path)
    train_df = pd.read_csv(train_data_path)
    
    pmhc_col = 'pMHC' if 'pMHC' in train_df.columns else 'epitope'
    tcr_col = 'beta' if 'beta' in train_df.columns else 'TCR'
    
    if pmhc_col not in train_df.columns:
        raise ValueError(f"Cannot find pMHC related column in training set, available columns: {train_df.columns.tolist()}")
    if tcr_col not in train_df.columns:
        raise ValueError(f"Cannot find TCR related column in training set, available columns: {train_df.columns.tolist()}")
    
    train_pmhc_counts = train_df.groupby(pmhc_col)[tcr_col].nunique().to_dict()
    
    test_categories = []
    test_pmhc_col = 'pMHC' if 'pMHC' in test_df.columns else 'epitope'
    
    if test_pmhc_col not in test_df.columns:
        raise ValueError(f"Cannot find pMHC related column in test set, available columns: {test_df.columns.tolist()}")
    
    for _, row in test_df.iterrows():
        test_pmhc = row[test_pmhc_col]
        
        if test_pmhc in train_pmhc_counts:
            tcr_count = train_pmhc_counts[test_pmhc]
            if tcr_count > GENERATION_CONFIG['normal_threshold']:
                category = 'normal'
            elif tcr_count >= GENERATION_CONFIG['fewshot_threshold']:
                category = 'fewshot'
            else:
                category = 'zeroshot'
        else:
            category = 'zeroshot'
        
        test_categories.append(category)
    
    category_counts = Counter(test_categories)
    total_samples = len(test_categories)
    
    category_stats = {
        'total_samples': total_samples,
        'normal_count': category_counts.get('normal', 0),
        'fewshot_count': category_counts.get('fewshot', 0),
        'zeroshot_count': category_counts.get('zeroshot', 0),
        'normal_ratio': category_counts.get('normal', 0) / total_samples,
        'fewshot_ratio': category_counts.get('fewshot', 0) / total_samples,
        'zeroshot_ratio': category_counts.get('zeroshot', 0) / total_samples,
        'train_pmhc_counts': train_pmhc_counts,
        'test_categories': test_categories
    }
    
    return test_categories, category_stats

def apply_sampling_constraints(logits, decoder_input, top_k, top_p, repetition_penalty):
    """Apply sampling constraints: repetition penalty + Top-k + Top-p filtering"""
    batch_size = logits.size(0)
    
    if repetition_penalty != 1.0:
        for i in range(batch_size):
            for token_id in set(decoder_input[i].tolist()):
                if token_id < logits.size(-1):
                    if logits[i, token_id] < 0:
                        logits[i, token_id] *= repetition_penalty
                    else:
                        logits[i, token_id] /= repetition_penalty
    
    probs = F.softmax(logits, dim=-1)
    
    if top_k > 0:
        top_k_values, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(1, top_k_indices, top_k_values)
        probs = probs_filtered
    
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
    
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    return torch.log(probs + 1e-8)

def decode_tokens_to_sequence(token_ids, tcr_idx2token):
    """Convert token sequence to string sequence"""
    sequence = []
    for token_id in token_ids.tolist():
        if token_id == 1:
            continue
        elif token_id == 2:
            break
        else:
            sequence.append(tcr_idx2token.get(token_id, '<UNK>'))
    return ''.join(sequence)


def ensure_reproducibility(seed: int):
    """Ensure reproducible results"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def softmax_np(x):
    """Numpy implementation of softmax"""
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def alpha_update(var_out, var_max, var_list, q):
    """
    Update alpha parameter (balance model confidence and deduplication weight)
    
    Args:
        var_out: Variance of current output distribution (uncertainty)
        var_max: Maximum possible variance (entropy of uniform distribution)
        var_list: List of historical variance
        q: Adaptive strength parameter
    
    Returns:
        Updated alpha value [0, 1]
    """
    if len(var_list) == 0:
        threshold = var_out
    else:
        threshold = np.median(var_list)
    
    var_centered = (var_out - threshold) / var_max
    sigmoid_arg = (q / 2) * (np.log((1 + var_centered) / (1 - var_centered + 1e-10)))
    return np.round(np.exp(sigmoid_arg) / (1 + np.exp(sigmoid_arg)), 2)

def k_update(var_out, var_max, var_list, q):
    """
    Update k parameter (candidate pool size)
    
    Args:
        var_out: Variance of current output distribution
        var_max: Maximum possible variance
        var_list: List of historical variance
        q: Adaptive strength parameter
    
    Returns:
        Updated k value [5, 15]
    """
    if len(var_list) == 0:
        threshold = var_out
    else:
        threshold = np.median(var_list)
    
    var_centered = (var_out - threshold) / var_max
    sigmoid_arg = (q / 2) * (np.log((1 + var_centered) / (1 - var_centered + 1e-10)))
    return int(np.round((np.exp(sigmoid_arg) / (1 + np.exp(sigmoid_arg))) * 10 + 5, 0))

def adaptive_k(logits, config):
    """
    Adaptively adjust k value based on logits uncertainty
    
    Args:
        logits: [1, vocab_size] or [vocab_size] logits
        config: Configuration object containing model.q parameter
    
    Returns:
        k: Adaptive candidate pool size
        var_out: Variance (entropy) of current distribution
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    vocab_size = logits.shape[1]
    uniform_dist = [1/vocab_size for _ in range(vocab_size)]
    
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    
    var_max = entropy(uniform_dist)
    
    var_out = entropy(probs)
    
    q = config.model.q if hasattr(config.model, 'q') else 8
    
    ratio = var_out / var_max
    k = int(5 + ratio * 10)
    k = max(5, min(15, k))
    
    return k, var_out

def adaptive_alpha(logits, beam_width, config):
    """
    Adaptively adjust alpha value based on logits uncertainty
    
    Args:
        logits: [1, vocab_size] or [vocab_size] logits
        beam_width: current beam width
        config: configuration object
    
    Returns:
        alpha: adaptive balance parameter [0, 1]
        var_out: variance of current distribution
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    vocab_size = logits.shape[1]
    uniform_dist = [1/vocab_size for _ in range(vocab_size)]
    
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    
    var_max = entropy(uniform_dist)
    var_out = entropy(probs)
    
    ratio = var_out / var_max
    alpha = 0.6 * (1 - ratio)
    alpha = max(0.1, min(0.8, alpha))
    
    return alpha, var_out

def ranking_fast_tcr(context_hidden, next_hidden, top_k_probs, alpha, beam_width):
    """
    Rank candidate tokens (combining probability and deduplication)
    
    Args:
        context_hidden: [1, seq_len, embed_dim] - representation of currently generated sequence
        next_hidden: [k, 1, embed_dim] - representation of k candidate tokens
        top_k_probs: [1, k] - probabilities of k candidate tokens
        alpha: balance parameter [0, 1]
        beam_width: k value
    
    Returns:
        selected_idx: selected candidate index (scalar)
    """
    norm_context = context_hidden / (context_hidden.norm(dim=2, keepdim=True) + 1e-10)
    norm_next = next_hidden / (next_hidden.norm(dim=2, keepdim=True) + 1e-10)
    
    cosine_matrix = torch.matmul(norm_next, norm_context.transpose(1, 2))
    
    degeneration_penalty = torch.max(cosine_matrix, dim=-1)[0].squeeze(-1)
    
    next_top_k_probs = top_k_probs.view(-1)
    
    scores = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    
    selected_idx = scores.argmax().item()
    
    return selected_idx

def apply_repetition_penalty_beam(logits, generated_tokens, repetition_penalty):
    """Apply repetition penalty for beam search (no top-k/top-p filtering)"""
    if repetition_penalty != 1.0:
        for token_id in set(generated_tokens):
            if token_id < logits.size(-1):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty
    return logits

def apply_ngram_blocking(logits, generated_tokens, no_repeat_ngram_size):
    """Block repeated n-grams"""
    if no_repeat_ngram_size <= 0 or len(generated_tokens) < no_repeat_ngram_size:
        return logits
    
    if len(generated_tokens) >= no_repeat_ngram_size - 1:
        ngram_prefix = generated_tokens[-(no_repeat_ngram_size-1):]
        
        for next_token in range(logits.size(-1)):
            candidate_ngram = ngram_prefix + [next_token]
            
            for i in range(len(generated_tokens) - no_repeat_ngram_size + 1):
                existing_ngram = generated_tokens[i:i + no_repeat_ngram_size]
                if candidate_ngram == existing_ngram:
                    logits[next_token] = -float('inf')
                    break
    
    return logits

def apply_min_length_constraint(logits, generated_tokens, min_length, sep_token_id=2):
    """Apply minimum length constraint"""
    if len(generated_tokens) < min_length:
        logits[sep_token_id] = -float('inf')
    return logits

def apply_diversity_penalty(logits, selected_tokens_in_group, diversity_penalty):
    """Apply diversity penalty within group"""
    if diversity_penalty > 0:
        for token_id, count in selected_tokens_in_group.items():
            if token_id < logits.size(-1):
                logits[token_id] -= diversity_penalty * count
    return logits

def conditional_beam_generation(model, enc_inputs, masked_pos, tcr_idx2token, max_length=32,
                               num_beams=8, num_return_sequences=5, length_penalty=1.0,
                               min_length=5, no_repeat_ngram_size=3, repetition_penalty=1.25,
                               diversity_penalty=0.0, num_beam_groups=1, ensure_unique=True,
                               min_edit_distance=0, device=None):
    """Conditional beam search generation"""
    model.eval()
    with torch.no_grad():
        batch_size = enc_inputs.size(0)
        
        enc_outputs_bert, _ = model.bert(enc_inputs, masked_pos)
        enc_outputs, _ = model.encoder(enc_inputs, enc_outputs_bert)
        
        all_results = []
        
        for sample_idx in range(batch_size):
            sample_enc_outputs = enc_outputs[sample_idx:sample_idx+1]
            sample_enc_inputs = enc_inputs[sample_idx:sample_idx+1]
            
            beams = [{'tokens': [1], 'score': 0.0}]
            finished = []
            finished_set = set()
            last_best_candidates = []
            
            group_size = num_beams // num_beam_groups
            
            for step in range(max_length - 1):
                if len(finished) >= num_return_sequences:
                    break
                
                candidates = []
                
                for group_idx in range(num_beam_groups):
                    group_start = group_idx * group_size
                    group_end = min(group_start + group_size, len(beams))
                    group_beams = beams[group_start:group_end]
                    
                    if not group_beams:
                        continue
                    
                    selected_tokens_in_group = {}
                    group_candidates = []
                    
                    for beam in group_beams:
                        tokens = beam['tokens']
                        score = beam['score']
                        
                        decoder_input = torch.tensor([tokens], dtype=torch.long).to(device)
                        
                        dec_outputs_gpt, _ = model.gpt(decoder_input)
                        dec_outputs, _, _ = model.decoder(
                            decoder_input, sample_enc_inputs, sample_enc_outputs, dec_outputs_gpt
                        )
                        dec_outputs = dec_outputs + dec_outputs_gpt
                        logits = model.projection(dec_outputs)[:, -1, :].squeeze(0)
                        
                        logits = apply_repetition_penalty_beam(logits, tokens, repetition_penalty)
                        logits = apply_ngram_blocking(logits, tokens, no_repeat_ngram_size)
                        logits = apply_min_length_constraint(logits, tokens, min_length)
                        
                        logits[0] = -float('inf')
                        
                        logits = apply_diversity_penalty(logits, selected_tokens_in_group, diversity_penalty)
                        
                        log_probs = F.log_softmax(logits, dim=-1)
                        
                        top_log_probs, top_indices = torch.topk(log_probs, min(group_size, log_probs.size(-1)))
                        
                        for prob, token_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                            new_score = score + prob
                            new_tokens = tokens + [token_id]
                            
                            group_candidates.append({
                                'tokens': new_tokens,
                                'score': new_score,
                                'token_id': token_id
                            })
                            
                            selected_tokens_in_group[token_id] = selected_tokens_in_group.get(token_id, 0) + 1
                    
                    group_candidates.sort(key=lambda x: x['score'], reverse=True)
                    candidates.extend(group_candidates[:group_size])
                
                new_beams = []
                
                for candidate in candidates:
                    tokens = candidate['tokens']
                    score = candidate['score']
                    token_id = candidate['token_id']
                    
                    if token_id == 2:
                        normalized_score = score / (len(tokens) ** length_penalty)
                        
                        seq_tuple = tuple(tokens)
                        if seq_tuple in finished_set:
                            continue
                        
                        if min_edit_distance > 0 and finished:
                            skip = False
                            candidate_seq = decode_tokens_to_sequence(torch.tensor(tokens), tcr_idx2token)
                            for finished_item in finished:
                                finished_seq = decode_tokens_to_sequence(torch.tensor(finished_item['tokens']), tcr_idx2token)
                                if Levenshtein.distance(candidate_seq, finished_seq) < min_edit_distance:
                                    skip = True
                                    break
                            if skip:
                                continue
                        
                        finished.append({
                            'tokens': tokens,
                            'score': normalized_score
                        })
                        finished_set.add(seq_tuple)
                        
                        if len(finished) >= num_return_sequences:
                            break
                    else:
                        new_beams.append(candidate)
                
                new_beams.sort(key=lambda x: x['score'], reverse=True)
                beams = new_beams[:num_beams]
                
                last_best_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:num_beams]
                
                if not beams:
                    break
            
            while len(finished) < num_return_sequences and beams:
                beam = beams.pop(0)
                tokens = beam['tokens']
                score = beam['score']
                
                normalized_score = score / (len(tokens) ** length_penalty)
                
                seq_tuple = tuple(tokens)
                if ensure_unique and seq_tuple in finished_set:
                    continue
                
                if min_edit_distance > 0 and finished:
                    skip = False
                    candidate_seq = decode_tokens_to_sequence(torch.tensor(tokens), tcr_idx2token)
                    for finished_item in finished:
                        finished_seq = decode_tokens_to_sequence(torch.tensor(finished_item['tokens']), tcr_idx2token)
                        if Levenshtein.distance(candidate_seq, finished_seq) < min_edit_distance:
                            skip = True
                            break
                    if skip:
                        continue
                
                finished.append({
                    'tokens': tokens,
                    'score': normalized_score
                })
                finished_set.add(seq_tuple)
            
            if len(finished) < num_return_sequences and not beams:
                for cand in last_best_candidates:
                    tokens = cand['tokens']
                    if tokens[-1] != 2:
                        tokens = tokens + [2]
                    seq_tuple = tuple(tokens)
                    if seq_tuple in finished_set:
                        continue
                    finished.append({'tokens': tokens, 'score': cand['score']/(len(tokens)**length_penalty)})
                    finished_set.add(seq_tuple)
                    if len(finished) >= num_return_sequences:
                        break

            finished.sort(key=lambda x: x['score'], reverse=True)
            sample_results = []
            
            for i in range(min(num_return_sequences, len(finished))):
                sequence = decode_tokens_to_sequence(torch.tensor(finished[i]['tokens']), tcr_idx2token)
                sample_results.append(sequence)
            
            while len(sample_results) < num_return_sequences:
                sample_results.append("")
            
            all_results.append(sample_results)
    
    return all_results


def hf_beam_generation(model, enc_inputs, masked_pos, tcr_idx2token, 
                       max_length=32,
                       num_beams=8, 
                       num_return_sequences=5, 
                       length_penalty=1.0,
                       min_length=5, 
                       no_repeat_ngram_size=3, 
                       repetition_penalty=1.25,
                       diversity_penalty=0.0,
                       num_beam_groups=1,
                       ensure_unique=True,
                       device=None):
    
    if device is None:
        device = enc_inputs.device
    
    enc_inputs = enc_inputs.to(device)
    masked_pos = masked_pos.to(device)
    
    model.eval()
    
    with torch.no_grad():
        batch_size = enc_inputs.size(0)
        vocab_size = len(tcr_idx2token)
        
        enc_outputs_bert, _ = model.bert(enc_inputs, masked_pos)
        enc_outputs, _ = model.encoder(enc_inputs, enc_outputs_bert)
        
        all_results = []
        
        for sample_idx in range(batch_size):
            sample_enc_outputs = enc_outputs[sample_idx:sample_idx+1]
            sample_enc_inputs = enc_inputs[sample_idx:sample_idx+1]
            
            expanded_enc_outputs = sample_enc_outputs.repeat_interleave(num_beams, dim=0)
            expanded_enc_inputs = sample_enc_inputs.repeat_interleave(num_beams, dim=0)
            
            beam_scorer = BeamSearchScorer(
                batch_size=1,
                num_beams=num_beams,
                device=device,
                length_penalty=length_penalty,
                do_early_stopping=True,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            
            logits_processor = LogitsProcessorList()
            
            if repetition_penalty != 1.0:
                logits_processor.append(
                    RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
                )
            
            if no_repeat_ngram_size > 0:
                logits_processor.append(
                    NoRepeatNGramLogitsProcessor(ngram_size=no_repeat_ngram_size)
                )
            
            if min_length > 0:
                eos_token_id_tensor = torch.tensor([2], dtype=torch.long, device=device)
                logits_processor.append(
                    MinLengthLogitsProcessor(
                        min_length=min_length,
                        eos_token_id=eos_token_id_tensor
                    )
                )
            
            if num_beam_groups > 1 and diversity_penalty > 0:
                logits_processor.append(
                    HammingDiversityLogitsProcessor(
                        diversity_penalty=diversity_penalty,
                        num_beams=num_beams,
                        num_beam_groups=num_beam_groups
                    )
                )
            
            input_ids = torch.full((num_beams, 1), 1, dtype=torch.long, device=device)
            
            beam_scores = torch.zeros((num_beams,), dtype=torch.float, device=device)
            beam_scores[1:] = -1e9
            
            for step in range(max_length - 1):
                input_ids = input_ids.to(device)
                
                dec_outputs_gpt, _ = model.gpt(input_ids)
                dec_outputs, _, _ = model.decoder(
                    input_ids, 
                    expanded_enc_inputs, 
                    expanded_enc_outputs, 
                    dec_outputs_gpt
                )
                dec_outputs = dec_outputs + dec_outputs_gpt
                logits = model.projection(dec_outputs)[:, -1, :]
                
                next_token_logits = logits_processor(input_ids, logits)
                
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                
                next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
                
                next_token_scores_reshaped = next_token_scores.view(1, num_beams * vocab_size)
                
                next_token_scores_topk, next_tokens = torch.topk(
                    next_token_scores_reshaped, 
                    2 * num_beams, 
                    dim=1, 
                    largest=True, 
                    sorted=True
                )
                
                next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
                next_tokens = next_tokens % vocab_size
                
                beam_outputs = beam_scorer.process(
                    input_ids.to(device),
                    next_token_scores_topk.to(device),
                    next_tokens.to(device),
                    next_indices.to(device),
                    pad_token_id=0,
                    eos_token_id=2,
                    beam_indices=None,
                )
                
                beam_scores = beam_outputs["next_beam_scores"].to(device)
                beam_next_tokens = beam_outputs["next_beam_tokens"].to(device)
                beam_idx = beam_outputs["next_beam_indices"].to(device)
                
                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1).to(device)
                
                if beam_scorer.is_done:
                    break
            
            sequence_outputs = beam_scorer.finalize(
                input_ids.to(device),
                beam_scores.to(device),
                next_tokens.to(device),
                next_indices.to(device),
                pad_token_id=0,
                eos_token_id=2,
                max_length=max_length,
                beam_indices=None,
            )
            
            generated_sequences = sequence_outputs["sequences"].cpu()
            generated_scores = sequence_outputs["sequence_scores"].cpu()
            
            if ensure_unique:
                unique_sequences = []
                seen_sequences = set()
                
                for seq, score in zip(generated_sequences, generated_scores):
                    seq_str = decode_tokens_to_sequence(seq, tcr_idx2token)
                    if seq_str not in seen_sequences:
                        unique_sequences.append(seq_str)
                        seen_sequences.add(seq_str)
                    
                    if len(unique_sequences) >= num_return_sequences:
                        break
                
                while len(unique_sequences) < num_return_sequences:
                    unique_sequences.append("")
                
                sample_results = unique_sequences
            else:
                sample_results = [
                    decode_tokens_to_sequence(seq, tcr_idx2token) 
                    for seq in generated_sequences[:num_return_sequences]
                ]
            
            all_results.append(sample_results)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return all_results


def decode_tokens_to_sequence(token_ids, tcr_idx2token):
    """Convert token sequence to string sequence"""
    sequence = []
    for token_id in token_ids.tolist():
        if token_id == 1:
            continue
        elif token_id == 2:
            break
        elif token_id == 0:
            break
        else:
            sequence.append(tcr_idx2token.get(token_id, '<UNK>'))
    return ''.join(sequence)


def tkns_generation(model, enc_inputs, masked_pos, tcr_idx2token, max_length=32,
                          top_p=None, top_k=None, repetition_penalty=None, device=None):
    """Pure tkns generation: use real pMHC conditions"""
    if top_p is None:
        top_p = GENERATION_CONFIG['sampling_params']['top_p']
    if top_k is None:
        top_k = GENERATION_CONFIG['sampling_params']['top_k'] 
    if repetition_penalty is None:
        repetition_penalty = GENERATION_CONFIG['sampling_params']['repetition_penalty']
    
    model.eval()
    with torch.no_grad():
        batch_size = enc_inputs.size(0)
        
        enc_outputs_bert, _ = model.bert(enc_inputs, masked_pos)
        enc_outputs, _ = model.encoder(enc_inputs, enc_outputs_bert)
        
        decoder_input = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)
        
        generated_sequences = []
        
        for step in range(max_length - 1):
            dec_outputs_gpt, _ = model.gpt(decoder_input)
            dec_outputs, _, _ = model.decoder(decoder_input, enc_inputs, enc_outputs, dec_outputs_gpt)
            dec_outputs = dec_outputs + dec_outputs_gpt
            logits = model.projection(dec_outputs)[:, -1, :]
            
            logits = apply_sampling_constraints(logits, decoder_input, top_k, top_p, repetition_penalty)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            
            if (next_token == 2).all():
                break
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        for i in range(batch_size):
            sequence = decode_tokens_to_sequence(decoder_input[i], tcr_idx2token)
            generated_sequences.append(sequence)
    
    return generated_sequences


def contrastive_decoding_one_step_tcr(model, enc_inputs, enc_outputs, decoder_input, 
                                      config, device, var_list, cached_dec_hidden=None):
    """
    Execute single-step adaptive contrastive decoding (optimized: reduce redundant computation)
    
    Args:
        model: GRA_Enhanced model
        enc_inputs: [1, seq_len] - pMHC encoding (already encoded, no change)
        enc_outputs: [1, seq_len, embed_dim] - pMHC encoder output (cached)
        decoder_input: [1, current_len] - currently generated TCR token sequence
        config: Configuration object (containing q parameter)
        device: Device
        var_list: Historical variance list (for adaptation)
        cached_dec_hidden: [1, current_len, embed_dim] - cached decoder hidden states (optional, for optimization)
    
    Returns:
        next_token_id: [1, 1] - next token
        last_hidden_states: [1, current_len+1, embed_dim] - updated hidden states
    """
    if cached_dec_hidden is not None:
        dec_outputs = cached_dec_hidden
        logits = model.projection(dec_outputs)[:, -1, :]
    else:
        dec_outputs_gpt, _ = model.gpt(decoder_input)
        dec_outputs, _, _ = model.decoder(decoder_input, enc_inputs, enc_outputs, dec_outputs_gpt)
        dec_outputs = dec_outputs + dec_outputs_gpt
        logits = model.projection(dec_outputs)[:, -1, :]
    
    beam_width, var_out = adaptive_k(logits, config)
    alpha, var_out_k = adaptive_alpha(logits, beam_width, config)
    
    var_list.append(var_out)
    
    next_probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(next_probs, k=beam_width, dim=-1)
    
    current_len = decoder_input.size(1)
    decoder_input_expanded = decoder_input.repeat(beam_width, 1)
    
    top_k_ids_expanded = top_k_ids.transpose(0, 1)
    candidate_seqs = torch.cat([decoder_input_expanded, top_k_ids_expanded], dim=1)
    
    enc_inputs_expanded = enc_inputs.repeat(beam_width, 1)
    enc_outputs_expanded = enc_outputs.repeat(beam_width, 1, 1)
    
    dec_outputs_gpt_batch, _ = model.gpt(candidate_seqs)
    dec_outputs_batch, _, _ = model.decoder(
        candidate_seqs, 
        enc_inputs_expanded, 
        enc_outputs_expanded, 
        dec_outputs_gpt_batch
    )
    dec_outputs_batch = dec_outputs_batch + dec_outputs_gpt_batch
    
    next_hidden = dec_outputs_batch[:, -1:, :]
    
    context_hidden = dec_outputs[:, :, :]
    
    selected_idx = ranking_fast_tcr(
        context_hidden,
        next_hidden,
        top_k_probs,
        alpha,
        beam_width
    )
    
    next_token_id = top_k_ids[:, selected_idx:selected_idx+1]
    
    selected_full_hidden = dec_outputs_batch[selected_idx:selected_idx+1, :, :]
    
    return next_token_id, selected_full_hidden


def acs_generation(model, enc_inputs, masked_pos, tcr_idx2token, 
                   max_length=32, initial_k=10, q=8, device=None):
    """
    Adaptive Contrastive Search generation (optimized: use hidden state cache)
    
    Args:
        model: GRA_Enhanced model
        enc_inputs: [batch_size, seq_len] - pMHC input
        masked_pos: mask positions
        tcr_idx2token: token dictionary
        max_length: maximum generation length
        initial_k: initial k value (will be adaptively adjusted)
        q: adaptive strength parameter
        device: device
    
    Returns:
        all_results: list of generated sequences
    """
    if device is None:
        device = enc_inputs.device
    
    model.eval()
    with torch.no_grad():
        batch_size = enc_inputs.size(0)
        
        enc_outputs_bert, _ = model.bert(enc_inputs, masked_pos)
        enc_outputs, _ = model.encoder(enc_inputs, enc_outputs_bert)
        
        all_results = []
        
        for sample_idx in range(batch_size):
            sample_enc_outputs = enc_outputs[sample_idx:sample_idx+1]
            sample_enc_inputs = enc_inputs[sample_idx:sample_idx+1]
            
            decoder_input = torch.tensor([[1]], dtype=torch.long, device=device)
            
            config = type('Config', (), {
                'model': type('Model', (), {'q': q})()
            })()
            
            var_list = []
            
            cached_dec_hidden = None
            
            for step in range(max_length - 1):
                next_token, cached_dec_hidden = contrastive_decoding_one_step_tcr(
                    model,
                    sample_enc_inputs,
                    sample_enc_outputs,
                    decoder_input,
                    config,
                    device,
                    var_list,
                    cached_dec_hidden
                )
                
                if next_token.item() == 2:
                    break
                
                decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            sequence = decode_tokens_to_sequence(decoder_input[0], tcr_idx2token)
            all_results.append(sequence)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return all_results


def setup_generation_logger(model_path, generation_mode):
    """Setup independent logger for generation phase"""
    model_save_dir = os.path.dirname(model_path)
    os.makedirs(model_save_dir, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'generation_{generation_mode}_{current_time}.log'
    logging_save_path = os.path.join(model_save_dir, log_filename)
    
    logger = logging.getLogger(f'generation_{generation_mode}')
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(logging_save_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, logging_save_path

def log_generation_config(logger, generation_mode, args=None):
    """Write current configuration parameters to generation log"""
    logger.info("=" * 60)
    logger.info("Current Configuration Parameters")
    logger.info("=" * 60)
    logger.info(f"Generation Mode: {generation_mode}")
    
    params = GENERATION_CONFIG['sampling_params']
    logger.info("\nSampling Parameters:")
    logger.info(f"  top_p: {params['top_p']}")
    logger.info(f"  top_k: {params['top_k']}")
    logger.info(f"  repetition_penalty: {params['repetition_penalty']}")
    
    if generation_mode == "conditional_beam" and args:
        logger.info("\nBeam Search Parameters:")
        logger.info(f"  num_beams: {args.num_beams}")
        logger.info(f"  num_return_sequences: {args.num_return_sequences}")
        logger.info(f"  length_penalty: {args.length_penalty}")
        logger.info(f"  min_length: {args.min_length}")
        logger.info(f"  no_repeat_ngram_size: {args.no_repeat_ngram_size}")
        logger.info(f"  diversity_penalty: {args.diversity_penalty}")
        logger.info(f"  num_beam_groups: {args.num_beam_groups}")
        logger.info(f"  ensure_unique: {args.ensure_unique}")
        logger.info(f"  min_edit_distance: {args.min_edit_distance}")
        logger.info(f"  use_hf_beam: {args.use_hf_beam}")
    
    if generation_mode == "acs" and args:
        logger.info("\nACS (Adaptive Contrastive Search) Parameters:")
        logger.info(f"  initial_k: {args.initial_k}")
        logger.info(f"  q (Adaptive Strength): {args.acs_q}")
        logger.info(f"  adaptive_alpha: Enabled")
        logger.info(f"  adaptive_k: Enabled")
        logger.info(f"  Note: k and alpha will be dynamically adjusted based on model uncertainty")
    
    if generation_mode == "ensemble" and args:
        logger.info("\nEnsemble Dual Ratio Control Parameters:")
        logger.info(f"  Methods: {args.ensemble_methods}")
        logger.info(f"  Generation ratios (not required to sum to 1): {args.generation_ratios}")
        if args.mixture_ratios and args.mixture_ratios.strip():
            logger.info(f"  Mixture ratios (must sum to 1): {args.mixture_ratios}")
        else:
            logger.info(f"  Mixture mode: Mix all and sort by score")
        logger.info(f"  Total return sequences: {args.num_return_sequences}")
        logger.info(f"  Unified scoring: Log probability (length normalized)")
        logger.info(f"  Description: Generation ratios control how many to generate, mixture ratios control final selection")

    logger.info("\nCategory Thresholds:")
    logger.info(f"  normal_threshold: {GENERATION_CONFIG['normal_threshold']}")
    logger.info(f"  fewshot_threshold: {GENERATION_CONFIG['fewshot_threshold']}")

def log_test_stratification_results(logger, category_stats):
    """Log test set stratification results"""
    logger.info("=" * 60)
    logger.info("Test Set Stratification Results")
    logger.info("=" * 60)
    
    total = category_stats['total_samples']
    logger.info(f"Total Samples: {total}")
    
    logger.info("\nSample Distribution:")
    logger.info(f"  Normal:   {category_stats['normal_count']:4d} samples ({category_stats['normal_ratio']:.1%})")
    logger.info(f"  Fewshot:  {category_stats['fewshot_count']:4d} samples ({category_stats['fewshot_ratio']:.1%})")
    logger.info(f"  Zeroshot: {category_stats['zeroshot_count']:4d} samples ({category_stats['zeroshot_ratio']:.1%})")
    
    logger.info(f"\nStratification Criteria:")
    logger.info(f"  Normal:   TCR count in training set > {GENERATION_CONFIG['normal_threshold']}")
    logger.info(f"  Fewshot:  TCR count in training set {GENERATION_CONFIG['fewshot_threshold']}-{GENERATION_CONFIG['normal_threshold']}")
    logger.info(f"  Zeroshot: TCR count in training set < {GENERATION_CONFIG['fewshot_threshold']} or not appeared")

def calculate_diversity(df):
    """Calculate diversity: how many unique TCRs generated per pMHC, calculate mean across all pMHCs"""
    diversity_by_pmhc = df.groupby('epitope')['beta'].nunique()
    return diversity_by_pmhc.mean()

def calculate_recovery_rate(generated_df, true_df):
    """
    Calculate sequence recovery rate:
    1. For each pMHC, calculate minimum levenshtein distance between all generated TCR sequences and true TCR sequences
    2. Calculate mean of minimum distances for each pMHC
    3. Calculate mean of means across all pMHCs in the test set
    """
    pmhc_recovery_rates = {}
    
    for epitope in generated_df['epitope'].unique():
        if epitope not in true_df['pMHC'].values:
            print(f"Errorrrrr!Epitope {epitope} not found in true data")
            continue
            
        gen_sequences = generated_df[generated_df['epitope'] == epitope]['beta'].tolist()
        true_sequences = true_df[true_df['pMHC'] == epitope]['beta'].tolist()
        
        if not gen_sequences or not true_sequences:
            print(f"Errorrrrr!No sequences found for epitope {epitope}")
            continue
        
        min_distances = []
        for gen_seq in gen_sequences:
            min_dist = min([Levenshtein.distance(gen_seq, true_seq) for true_seq in true_sequences])
            min_dist = 1-(min_dist / len(gen_seq))
            min_distances.append(min_dist)
        
        avg_min_distance = np.mean(min_distances)
        pmhc_recovery_rates[epitope] = avg_min_distance
    
    overall_recovery_rate = np.mean(list(pmhc_recovery_rates.values())) if pmhc_recovery_rates else float('inf')
    
    return {
        'pmhc_recovery_rates': pmhc_recovery_rates,
        'overall_recovery_rate': overall_recovery_rate
    }

def calculate_unique_sequence_stats(df):
    """Calculate unique sequence statistics"""
    all_sequences = df['beta'].tolist()
    total_sequences = len(all_sequences)
    unique_all_sequences = set(all_sequences)
    unique_all_count = len(unique_all_sequences)
    unique_all_ratio = unique_all_count / total_sequences if total_sequences > 0 else 0
    
    unique_by_category = set()
    category_unique_stats = {}
    
    for category in ['normal', 'fewshot', 'zeroshot']:
        category_df = df[df['sample_category'] == category]
        if len(category_df) > 0:
            category_sequences = category_df['beta'].tolist()
            category_unique = set(category_sequences)
            category_unique_stats[category] = {
                'total': len(category_sequences),
                'unique': len(category_unique),
                'ratio': len(category_unique) / len(category_sequences)
            }
            unique_by_category.update(category_unique)
        else:
            category_unique_stats[category] = {'total': 0, 'unique': 0, 'ratio': 0}
    
    merged_unique_count = len(unique_by_category)
    merged_unique_ratio = merged_unique_count / total_sequences if total_sequences > 0 else 0
    
    return {
        'total_sequences': total_sequences,
        'unique_all_count': unique_all_count,
        'unique_all_ratio': unique_all_ratio,
        'merged_unique_count': merged_unique_count,
        'merged_unique_ratio': merged_unique_ratio,
        'category_stats': category_unique_stats
    }

def log_generation_results(logger, df, generation_mode):
    """Write generation result statistics to log"""
    logger.info("=" * 60)
    logger.info("Generation Results Statistics")
    logger.info("=" * 60)
    logger.info(f"Generation Mode: {generation_mode}")
    logger.info(f"Total Sequences: {len(df)}")
    
    category_stats = df['sample_category'].value_counts()
    logger.info("\nStatistics by Sample Category:")
    for category in ['normal', 'fewshot', 'zeroshot']:
        count = category_stats.get(category, 0)
        ratio = count / len(df) if len(df) > 0 else 0
        logger.info(f"  {category:12}: {count:4d} sequences ({ratio:.1%})")
    
    unique_stats = calculate_unique_sequence_stats(df)
    logger.info("\n" + "=" * 60)
    logger.info("Unique Sequence Statistics")
    logger.info("=" * 60)
    
    logger.info("1. Overall Sequence Uniqueness:")
    logger.info(f"   Total Sequences: {unique_stats['total_sequences']}")
    logger.info(f"   Unique Sequences: {unique_stats['unique_all_count']}")
    logger.info(f"   Unique Sequence Ratio: {unique_stats['unique_all_ratio']:.1%}")
    
    logger.info("\n2. Statistics After Deduplication and Merging by Category:")
    logger.info(f"   Merged Unique Sequences: {unique_stats['merged_unique_count']}")
    logger.info(f"   Merged Unique Ratio to Total: {unique_stats['merged_unique_ratio']:.1%}")
    
    logger.info("\n3. Uniqueness Within Each Category:")
    for category in ['normal', 'fewshot', 'zeroshot']:
        stats = unique_stats['category_stats'][category]
        if stats['total'] > 0:
            logger.info(f"   {category:12}: {stats['unique']}/{stats['total']} ({stats['ratio']:.1%} unique)")
        else:
            logger.info(f"   {category:12}: 0/0 (0.0% unique)")
    
    logger.info("\n" + "=" * 60)
    logger.info("Additional Statistics")
    logger.info("=" * 60)
    
    seq_lengths = df['beta'].apply(len)
    logger.info(f"Sequence Length Statistics:")
    logger.info(f"  Average Length: {seq_lengths.mean():.1f}")
    logger.info(f"  Minimum Length: {seq_lengths.min()}")
    logger.info(f"  Maximum Length: {seq_lengths.max()}")
    logger.info(f"  Length Std Dev: {seq_lengths.std():.1f}")
    
    logger.info("\nSequence Length by Category:")
    for category in ['normal', 'fewshot', 'zeroshot']:
        category_df = df[df['sample_category'] == category]
        if len(category_df) > 0:
            cat_lengths = category_df['beta'].apply(len)
            logger.info(f"  {category:12}: Avg={cat_lengths.mean():.1f}, Range=[{cat_lengths.min()}-{cat_lengths.max()}]")

def create_result_path(data_path, model_path, num_generated_per_sample, generation_mode="conditional_beam"):
    """Create result file path, select different directory based on generation mode"""
    test_name = os.path.splitext(os.path.basename(data_path))[0]
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    model_dir = os.path.dirname(model_path)
    if generation_mode == "acs":
        result_dir = os.path.join(model_dir, "result_acs")
    elif generation_mode == "ensemble":
        result_dir = os.path.join(model_dir, "result_ensemble")
    else:
        result_dir = os.path.join(model_dir, "result_beam")
    
    os.makedirs(result_dir, exist_ok=True)
    
    filename = f"{test_name}_{model_name}_gen{num_generated_per_sample}.csv"
    result_path = os.path.join(result_dir, filename)
    
    return result_path

def save_experiment_results_with_metrics(model_save_dir, results_data, model_name, test_name, generation_mode, df, true_tcr_df=None):
    """Save experiment results to CSV file, including recovery rate and diversity metrics, select directory based on generation mode"""
    diversity = calculate_diversity(df)
    results_data['diversity'] = diversity
    
    if true_tcr_df is not None and len(true_tcr_df) > 0:
        recovery_result = calculate_recovery_rate(df, true_tcr_df)
        results_data['overall_recovery_rate'] = recovery_result['overall_recovery_rate']
        
        pmhc_recovery_rates = recovery_result['pmhc_recovery_rates']
        for epitope, rate in pmhc_recovery_rates.items():
            safe_epitope = str(epitope).replace(',', '_').replace(' ', '_').replace('/', '_')
            results_data[f'recovery_rate_{safe_epitope}'] = rate
        
        results_data['num_pmhc_evaluated'] = len(pmhc_recovery_rates)
    else:
        print(f"Errorrrrr!No true TCR data found")
        results_data['overall_recovery_rate'] = float('inf')
        results_data['num_pmhc_evaluated'] = 0
    
    if generation_mode == "acs":
        result_dir = os.path.join(model_save_dir, "result_acs")
    elif generation_mode == "ensemble":
        result_dir = os.path.join(model_save_dir, "result_ensemble")
    else:
        result_dir = os.path.join(model_save_dir, "result_beam")
    
    os.makedirs(result_dir, exist_ok=True)
    
    results_filename = f'{test_name}_{model_name}_{generation_mode}_results.csv'
    results_path = os.path.join(result_dir, results_filename)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data['model_name'] = model_name
    results_data['test_name'] = test_name
    results_data['generation_mode'] = generation_mode
    results_data['timestamp'] = current_time
    
    results_df = pd.DataFrame([results_data])
    results_df_transposed = results_df.T
    results_df_transposed.columns = ['value']
    results_df_transposed.to_csv(results_path, header=True)
    return results_path

def calculate_sequence_log_probability(model, enc_inputs, enc_outputs, sequence_tokens, 
                                       tcr_token2idx, length_penalty=1.0, device=None):
    """
    Calculate log probability of sequence (unified scoring standard, applicable to all sampling methods)
    
    Args:
        model: GRA_Enhanced model
        enc_inputs: [1, seq_len] - pMHCencoding
        enc_outputs: [1, seq_len, embed_dim] - encoderoutput
        sequence_tokens: list[int] or str - sequence tokens or string
        tcr_token2idx: token to index mapping dictionary
        length_penalty: length penalty parameter
        device: device
    
    Returns:
        dict: {
            'log_prob': cumulative log probability,
            'normalized_score': length-normalized score,
            'perplexity': perplexity,
            'length': sequence length
        }
    """
    if device is None:
        device = enc_inputs.device
    
    if isinstance(sequence_tokens, str):
        tokens = [1]
        for char in sequence_tokens:
            token_id = tcr_token2idx.get(char, None)
            if token_id is not None:
                tokens.append(token_id)
        tokens.append(2)
        sequence_tokens = tokens
    
    if sequence_tokens[0] != 1:
        sequence_tokens = [1] + sequence_tokens
    
    if sequence_tokens[-1] != 2:
        sequence_tokens = sequence_tokens + [2]
    
    model.eval()
    with torch.no_grad():
        decoder_input = torch.tensor([sequence_tokens], dtype=torch.long, device=device)
        
        dec_outputs_gpt, _ = model.gpt(decoder_input)
        dec_outputs, _, _ = model.decoder(decoder_input, enc_inputs, enc_outputs, dec_outputs_gpt)
        dec_outputs = dec_outputs + dec_outputs_gpt
        logits = model.projection(dec_outputs)
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        total_log_prob = 0.0
        token_count = 0
        
        for i in range(1, len(sequence_tokens)):
            token_id = sequence_tokens[i]
            if token_id == 2:
                break
            if token_id == 0:
                continue
            
            token_log_prob = log_probs[0, i-1, token_id].item()
            total_log_prob += token_log_prob
            token_count += 1
        
        if token_count > 0:
            normalized_score = total_log_prob / (token_count ** length_penalty)
            perplexity = np.exp(-total_log_prob / token_count)
        else:
            normalized_score = -float('inf')
            perplexity = float('inf')
        
        return {
            'log_prob': total_log_prob,
            'normalized_score': normalized_score,
            'perplexity': perplexity,
            'length': token_count
        }


def ensemble_generation(model, enc_inputs, masked_pos, tcr_idx2token, tcr_token2idx,
                       args, device, methods=None, generation_ratios=None, mixture_ratios=None):
    """
    Ensemble multiple sampling methods, dual-ratio control version
    
    Args:
        model: GRA_Enhanced model
        enc_inputs: [batch_size, seq_len] - pMHC input
        masked_pos: mask positions
        tcr_idx2token: index to token mapping
        tcr_token2idx: token to index mapping
        args: argument object
        device: device
        methods: list of sampling methods
        generation_ratios: generation ratios (don't need to sum to 1)
        mixture_ratios: mixture ratios (optional, must sum to 1 if set)
    
    Returns:
        all_results: list of generated sequences
        all_method_stats: method statistics
    """
    if methods is None:
        methods = args.ensemble_methods.split(',')
    
    if generation_ratios is None:
        generation_ratios = [float(r) for r in args.generation_ratios.split(',')]
    
    use_mixture_ratios = False
    if mixture_ratios is None and args.mixture_ratios and args.mixture_ratios.strip():
        mixture_ratios = [float(r) for r in args.mixture_ratios.split(',')]
        use_mixture_ratios = True
    elif mixture_ratios is not None:
        use_mixture_ratios = True
    
    assert len(methods) == len(generation_ratios), \
        "Number of methods must match number of generation ratios"
    
    if use_mixture_ratios:
        assert len(methods) == len(mixture_ratios), \
            "Number of methods must match number of mixture ratios"
        assert abs(sum(mixture_ratios) - 1.0) < 1e-6, \
            f"Mixture ratios must sum to 1.0, current sum: {sum(mixture_ratios)}"
    
    batch_size = enc_inputs.size(0)
    total_sequences = args.num_return_sequences
    
    generation_counts = {}
    for method, ratio in zip(methods, generation_ratios):
        count = math.ceil(total_sequences * ratio)
        generation_counts[method] = count
    
    mixture_counts = None
    if use_mixture_ratios:
        mixture_counts = {}
        remaining = total_sequences
        for i, (method, ratio) in enumerate(zip(methods, mixture_ratios)):
            if i < len(methods) - 1:
                count = math.ceil(total_sequences * ratio)
                mixture_counts[method] = count
                remaining -= count
            else:
                mixture_counts[method] = remaining
    
    model.eval()
    with torch.no_grad():
        enc_outputs_bert, _ = model.bert(enc_inputs, masked_pos)
        enc_outputs, _ = model.encoder(enc_inputs, enc_outputs_bert)
    
    all_results = []
    all_method_stats = []
    
    for sample_idx in range(batch_size):
        sample_enc_outputs = enc_outputs[sample_idx:sample_idx+1]
        sample_enc_inputs = enc_inputs[sample_idx:sample_idx+1]
        
        method_sequences = {method: [] for method in methods}
        
        for method, count in generation_counts.items():
            if count <= 0:
                continue
            
            
            if method == 'beam':
                if args.use_hf_beam:
                    beam_results = hf_beam_generation(
                        model, sample_enc_inputs, masked_pos[sample_idx:sample_idx+1], 
                        tcr_idx2token,
                        max_length=args.maxlen,
                        num_beams=max(count * 2, args.num_beams),
                        num_return_sequences=count,
                        length_penalty=args.length_penalty,
                        min_length=args.min_length,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        repetition_penalty=args.repetition_penalty,
                        diversity_penalty=args.diversity_penalty,
                        num_beam_groups=args.num_beam_groups,
                        ensure_unique=False,
                        device=device
                    )
                else:
                    beam_results = conditional_beam_generation(
                        model, sample_enc_inputs, masked_pos[sample_idx:sample_idx+1], 
                        tcr_idx2token,
                        max_length=args.maxlen,
                        num_beams=max(count * 2, args.num_beams),
                        num_return_sequences=count,
                        length_penalty=args.length_penalty,
                        min_length=args.min_length,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        repetition_penalty=args.repetition_penalty,
                        diversity_penalty=args.diversity_penalty,
                        num_beam_groups=args.num_beam_groups,
                        ensure_unique=False,
                        min_edit_distance=0,
                        device=device
                    )
                
                for seq in beam_results[0]:
                    if seq and seq.strip():
                        method_sequences[method].append({
                            'sequence': seq,
                            'method': method
                        })
            
            elif method == 'acs':
                for _ in range(count):
                    acs_results = acs_generation(
                        model, sample_enc_inputs, masked_pos[sample_idx:sample_idx+1],
                        tcr_idx2token,
                        max_length=args.maxlen,
                        initial_k=args.initial_k,
                        q=args.acs_q,
                        device=device
                    )
                    for seq in acs_results:
                        if seq and seq.strip():
                            method_sequences[method].append({
                                'sequence': seq,
                                'method': method
                            })
            
            elif method == 'tkns':
                for _ in range(count):
                    cond_results = tkns_generation(
                        model, sample_enc_inputs, masked_pos[sample_idx:sample_idx+1],
                        tcr_idx2token,
                        max_length=args.maxlen,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        device=device
                    )
                    for seq in cond_results:
                        if seq and seq.strip():
                            method_sequences[method].append({
                                'sequence': seq,
                                'method': method
                            })
        
        for method, seqs in method_sequences.items():
            for item in seqs:
                score_info = calculate_sequence_log_probability(
                    model, sample_enc_inputs, sample_enc_outputs, 
                    item['sequence'], tcr_token2idx,
                    length_penalty=args.length_penalty,
                    device=device
                )
                item.update(score_info)
        
        for method in methods:
            seqs = method_sequences[method]
            
            unique_seqs = {}
            for item in seqs:
                seq = item['sequence']
                if seq not in unique_seqs:
                    unique_seqs[seq] = item
                else:
                    if item['normalized_score'] > unique_seqs[seq]['normalized_score']:
                        unique_seqs[seq] = item
            
            method_sequences[method] = sorted(
                unique_seqs.values(),
                key=lambda x: x['normalized_score'],
                reverse=True
            )
        
        if use_mixture_ratios:
            selected_sequences = []
            for method, count in mixture_counts.items():
                available = method_sequences[method]
                selected = available[:count]
                selected_sequences.extend(selected)
            
            selected_sequences.sort(key=lambda x: x['normalized_score'], reverse=True)
            
            final_sequences = selected_sequences[:total_sequences]
        else:
            all_sequences = []
            for method, seqs in method_sequences.items():
                all_sequences.extend(seqs)
            
            all_sequences.sort(key=lambda x: x['normalized_score'], reverse=True)
            
            final_sequences = all_sequences[:total_sequences]
        
        method_counter = Counter([item['method'] for item in final_sequences])
        method_stats = {
            method: method_counter.get(method, 0) / len(final_sequences) if final_sequences else 0
            for method in methods
        }
        all_method_stats.append(method_stats)
        
        sample_results = [item['sequence'] for item in final_sequences]
        
        while len(sample_results) < total_sequences:
            sample_results.append("")
        
        all_results.append(sample_results)
        
    
    return all_results, all_method_stats


def load_model_with_compatibility(model, model_path, device):
    """
    Compatible model loading function: handle models with or without null_condition parameter
    
    Args:
        model: model instance to load
        model_path: model file path
        device: device
    
    Returns:
        loaded model
    """
    state_dict = torch.load(model_path)
    
    if 'null_condition' not in state_dict:
        print("Warning: No null_condition parameter in model file, will use default value")
        model.null_condition.data = torch.randn(768)
        current_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in current_state_dict}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    
    model.to(device)
    return model

def create_parser():
    parser = argparse.ArgumentParser(description="GRA Generation Mode Optimized",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",dest="data_path",type=str,help="The data file in .csv format.",required=True)
    parser.add_argument("--train_data_path",dest="train_data_path",type=str,help="Training data path for test stratification.",required=False, default="")
    parser.add_argument("--model_path",dest="model_path",type=str, help="the path to load model.", required=True)
    parser.add_argument("--result_path",dest="result_path",type=str, default="",help="the path to store result.", required=False)
    parser.add_argument("--maxlen",dest="maxlen",type=int,help="maxlen of seq",default=32, required=False)
    parser.add_argument("--batch_size",dest="batch_size",type=int,help="batch_size",default=1, required=False)
    parser.add_argument("--random_seed",type=int, dest="random_seed", default=1, help="seed for reproductbility",required=False)
    
    parser.add_argument("--generation_mode",dest="generation_mode",type=str,default="conditional_beam",
                    choices=["tkns", "conditional_beam", "acs", "ensemble"],
                    help="Generation mode")

    parser.add_argument("--num_beams",dest="num_beams",type=int,default=200,
                    help="Number of beams for beam search")
    parser.add_argument("--num_return_sequences",dest="num_return_sequences",type=int,default=100,
                    help="Number of sequences to return per sample (must be <= num_beams)")
    parser.add_argument("--length_penalty",dest="length_penalty",type=float,default=1.0,
                    help="Length penalty for beam search")
    parser.add_argument("--min_length",dest="min_length",type=int,default=5,
                    help="Minimum length of generated sequences")
    parser.add_argument("--no_repeat_ngram_size",dest="no_repeat_ngram_size",type=int,default=3,
                    help="Size of n-grams to prevent repetition")

    parser.add_argument("--diversity_penalty",dest="diversity_penalty",type=float,default=0.0,
                    help="Diversity penalty for diverse beam search")
    parser.add_argument("--num_beam_groups",dest="num_beam_groups",type=int,default=1,
                    help="Number of beam groups for diverse beam search")

    parser.add_argument("--ensure_unique",action="store_true",
                    help="Ensure unique sequences in output")
    parser.add_argument("--min_edit_distance",dest="min_edit_distance",type=int,default=0,
                    help="Minimum edit distance between sequences (0 to disable)")
    
    parser.add_argument("--use_hf_beam",action="store_true",
                    help="Use HuggingFace BeamSearchScorer for faster beam search (5-10x speedup)")

    parser.add_argument("--initial_k",dest="initial_k",type=int,default=10,
                    help="Initial k value for ACS (will be adaptively adjusted)")
    parser.add_argument("--acs_q",dest="acs_q",type=int,default=8,
                    help="Q parameter for ACS adaptive strength (higher = more aggressive adaptation)")

    parser.add_argument("--ensemble_methods",dest="ensemble_methods",type=str,default="beam,acs,tkns",
                    help="Comma-separated list of methods for ensemble (e.g., 'beam,acs,tkns')")
    parser.add_argument("--generation_ratios",dest="generation_ratios",type=str,default="0.6,0.4,0.4",
                    help="Generation ratios for each method (not required to sum to 1, e.g., '0.6,0.4,0.4')")
    parser.add_argument("--mixture_ratios",dest="mixture_ratios",type=str,default="",
                    help="Mixture ratios for final selection (optional, must sum to 1 if set, e.g., '0.5,0.3,0.2'). If not set, all sequences are mixed and sorted.")

    parser.add_argument("--num_generated_per_sample",dest="num_generated_per_sample",type=int,default=100,
                       help="Number of sequences to generate per sample")
    
    parser.add_argument("--top_p",dest="top_p",type=float,default=0.92,help="Top-p sampling parameter")
    parser.add_argument("--top_k",dest="top_k",type=int,default=60,help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty",dest="repetition_penalty",type=float,default=1.25,help="Repetition penalty")
    
    
    parser.add_argument("--true_tcr_path",dest="true_tcr_path",type=str,default="",
                       help="Path to true TCR sequences for recovery rate calculation (default: same as data_path)")
    
    args = parser.parse_args()
    
    return args

def main(args):
    GENERATION_CONFIG['sampling_params']['top_p'] = args.top_p
    GENERATION_CONFIG['sampling_params']['top_k'] = args.top_k
    GENERATION_CONFIG['sampling_params']['repetition_penalty'] = args.repetition_penalty
    
    device = torch.device('cuda:0')
    set_seed(args.random_seed)
    ensure_reproducibility(args.random_seed)
    
    data = pd.read_csv(args.data_path)
    dataset, tcr_idx2token, pep_idx2token, vocab_size_tcr, vocab_size_pep = make_data_for_gra(
        data, 32, 55, 'generate')
    
    true_tcr_df = None
    if args.true_tcr_path:
        if os.path.exists(args.true_tcr_path):
            true_tcr_df = pd.read_csv(args.true_tcr_path)
        else:
            print(f"Warning: True TCR path {args.true_tcr_path} not found")
    else:
        if os.path.exists(args.data_path):
            true_tcr_df = pd.read_csv(args.data_path)
    
    test_categories = None
    category_stats = None
    if args.train_data_path and os.path.exists(args.train_data_path):
        test_categories, category_stats = categorize_test_samples_by_training_set(
            args.data_path, args.train_data_path)
    else:
        test_categories = ['normal'] * len(dataset)
        category_stats = {
            'total_samples': len(dataset),
            'normal_count': len(dataset),
            'fewshot_count': 0,
            'zeroshot_count': 0,
            'normal_ratio': 1.0,
            'fewshot_ratio': 0.0,
            'zeroshot_ratio': 0.0
        }
    
    data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    gen_logger, log_path = setup_generation_logger(args.model_path, args.generation_mode)
    
    log_generation_config(gen_logger, args.generation_mode, args)
    
    if category_stats:
        log_test_stratification_results(gen_logger, category_stats)
    
    if true_tcr_df is not None:
        gen_logger.info("=" * 60)
        gen_logger.info("True TCR Data Information")
        gen_logger.info("=" * 60)
        gen_logger.info(f"True TCR Data Path: {args.true_tcr_path if args.true_tcr_path else args.data_path}")
        gen_logger.info(f"True TCR Sequences: {len(true_tcr_df)}")
        gen_logger.info(f"True TCR Involving pMHC Count: {true_tcr_df['pMHC'].nunique() if 'pMHC' in true_tcr_df.columns else 'epitope'}")
    
    bert = BERT(vocab_size_pep)
    gpt = GPT_Model(vocab_size_tcr)
    model = GRA_Enhanced(bert, gpt, vocab_size_tcr)
    
    model = load_model_with_compatibility(model, args.model_path, device)
    
    if args.result_path == "":
        result_path = create_result_path(args.data_path, args.model_path, args.num_generated_per_sample, args.generation_mode)
    else:
        result_path = args.result_path
    
    if os.path.exists(result_path):
        print(f"Result file already exists: {result_path}")
        print("Skipping generation.")
        return
    
    results_data = {
        'generation_mode': args.generation_mode,
        'num_generated_per_sample': args.num_generated_per_sample,
        'normal_threshold': GENERATION_CONFIG['normal_threshold'],
        'fewshot_threshold': GENERATION_CONFIG['fewshot_threshold'],
        'top_p': GENERATION_CONFIG['sampling_params']['top_p'],
        'top_k': GENERATION_CONFIG['sampling_params']['top_k'],
        'repetition_penalty': GENERATION_CONFIG['sampling_params']['repetition_penalty'],
    }

    if args.generation_mode == "conditional_beam":
        results_data.update({
            'num_beams': args.num_beams,
            'num_return_sequences': args.num_return_sequences,
            'length_penalty': args.length_penalty,
            'min_length': args.min_length,
            'no_repeat_ngram_size': args.no_repeat_ngram_size,
            'diversity_penalty': args.diversity_penalty,
            'num_beam_groups': args.num_beam_groups,
            'ensure_unique': args.ensure_unique,
            'min_edit_distance': args.min_edit_distance,
            'use_hf_beam': args.use_hf_beam,
        })
    elif args.generation_mode == "acs":
        results_data.update({
            'initial_k': args.initial_k,
            'acs_q': args.acs_q,
            'adaptive_alpha': True,
            'adaptive_k': True,
        })
    elif args.generation_mode == "ensemble":
        results_data.update({
            'ensemble_methods': args.ensemble_methods,
            'generation_ratios': args.generation_ratios,
            'mixture_ratios': args.mixture_ratios,
            'num_return_sequences': args.num_return_sequences,
            'length_penalty': args.length_penalty,
            'unified_scoring': 'log_probability',
        })
    
    tcr_token2idx = {v: k for k, v in tcr_idx2token.items()}
    
    seq_list = []
    
    gen_logger.info("=" * 60)
    gen_logger.info("Starting sequence generation...")
    gen_logger.info("=" * 60)
    gen_logger.info(f"Sequences per Sample: {args.num_generated_per_sample}")
    
    print(f"\n=== Generation Mode: {args.generation_mode} ===")
    print(f"Generation ratios: {args.generation_ratios} ")
    print(f"Mixture ratios: {args.mixture_ratios} ")
    print(f"Target total sequences: {args.num_return_sequences}")


    with torch.no_grad():
        for batch_idx, (inputIds, mask) in enumerate(tqdm(data_loader)):
            inputIds = inputIds.to(device)
            mask = mask.to(device)
            
            batch_start_idx = batch_idx * args.batch_size
            sample_categories = []
            for i in range(inputIds.size(0)):
                sample_idx = batch_start_idx + i
                if sample_idx < len(test_categories):
                    category = test_categories[sample_idx]
                else:
                    category = 'normal'
                sample_categories.append(category)
            
            if args.generation_mode == "tkns":
                for gen_round in range(args.num_generated_per_sample):
                    generated_sequences = tkns_generation(
                        model, inputIds, mask, tcr_idx2token, 
                        max_length=args.maxlen, device=device
                    )
                    
                    for i in range(len(generated_sequences)):
                        pep_seq = inputIds[i].cpu().tolist()
                        pep = ''
                        for value in pep_seq:
                            if value == 0:
                                break
                            elif value in (1, 2):
                                continue
                            else:
                                pep = pep + pep_idx2token[value]
                        
                        category = sample_categories[i] if i < len(sample_categories) else 'normal'
                        seq_list.append([pep, generated_sequences[i], category])
            
            elif args.generation_mode == "acs":
                for gen_round in range(args.num_generated_per_sample):
                    generated_sequences = acs_generation(
                        model, inputIds, mask, tcr_idx2token,
                        max_length=args.maxlen,
                        initial_k=args.initial_k,
                        q=args.acs_q,
                        device=device
                    )
                    
                    for i in range(len(generated_sequences)):
                        pep_seq = inputIds[i].cpu().tolist()
                        pep = ''
                        for value in pep_seq:
                            if value == 0:
                                break
                            elif value in (1, 2):
                                continue
                            else:
                                pep = pep + pep_idx2token[value]
                        
                        category = sample_categories[i] if i < len(sample_categories) else 'normal'
                        seq_list.append([pep, generated_sequences[i], category])
            
            elif args.generation_mode == "ensemble":
                generated_sequences, method_stats = ensemble_generation(
                    model, inputIds, mask, tcr_idx2token, tcr_token2idx,
                    args, device
                )
                
                if not hasattr(args, '_ensemble_method_stats'):
                    args._ensemble_method_stats = []
                args._ensemble_method_stats.extend(method_stats)
                
                for i in range(len(generated_sequences)):
                    pep_seq = inputIds[i].cpu().tolist()
                    pep = ''
                    for value in pep_seq:
                        if value == 0:
                            break
                        elif value in (1, 2):
                            continue
                        else:
                            pep = pep + pep_idx2token[value]
                    
                    category = sample_categories[i] if i < len(sample_categories) else 'normal'
                    for seq in generated_sequences[i]:
                        if seq and seq.strip():
                            seq_list.append([pep, seq, category])
            
            elif args.generation_mode == "conditional_beam":
                if args.use_hf_beam:
                    generated_sequences = hf_beam_generation(
                        model, inputIds, mask, tcr_idx2token,
                        max_length=args.maxlen,
                        num_beams=args.num_beams,
                        num_return_sequences=args.num_return_sequences,
                        length_penalty=args.length_penalty,
                        min_length=args.min_length,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        repetition_penalty=args.repetition_penalty,
                        diversity_penalty=args.diversity_penalty,
                        num_beam_groups=args.num_beam_groups,
                        ensure_unique=args.ensure_unique,
                        device=device
                    )
                else:
                    generated_sequences = conditional_beam_generation(
                        model, inputIds, mask, tcr_idx2token,
                        max_length=args.maxlen,
                        num_beams=args.num_beams,
                        num_return_sequences=args.num_return_sequences,
                        length_penalty=args.length_penalty,
                        min_length=args.min_length,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        repetition_penalty=args.repetition_penalty,
                        diversity_penalty=args.diversity_penalty,
                        num_beam_groups=args.num_beam_groups,
                        ensure_unique=args.ensure_unique,
                        min_edit_distance=args.min_edit_distance,
                        device=device
                    )        
                for i in range(len(generated_sequences)):
                    pep_seq = inputIds[i].cpu().tolist()
                    pep = ''
                    for value in pep_seq:
                        if value == 0:
                            break
                        elif value == 1 or value == 2:
                            pass
                        else:
                            pep = pep + pep_idx2token[value]
                    
                    category = sample_categories[i] if i < len(sample_categories) else 'normal'
                    for seq in generated_sequences[i]:
                        if seq.strip():
                            seq_list.append([pep, seq, category])
    
    df = pd.DataFrame(seq_list, columns=['epitope', 'beta', 'sample_category'])
    df.to_csv(result_path, index=False)
    
    log_generation_results(gen_logger, df, args.generation_mode)
    
    sequence_recovery_rate = float('inf')
    if true_tcr_df is not None and len(true_tcr_df) > 0:
        recovery_result = calculate_recovery_rate(df, true_tcr_df)
        sequence_recovery_rate = recovery_result['overall_recovery_rate']
        pmhc_recovery_rates = recovery_result['pmhc_recovery_rates']
        
        gen_logger.info("=" * 60)
        gen_logger.info("Sequence Recovery Rate Results")
        gen_logger.info("=" * 60)
        gen_logger.info(f"Overall Sequence Recovery Rate (Average Min Edit Distance): {sequence_recovery_rate:.4f}")
        gen_logger.info(f"Number of pMHCs Evaluated: {len(pmhc_recovery_rates)}")
        gen_logger.info("Note: Lower values indicate generated sequences are more similar to true sequences")
        
        gen_logger.info("\nTop 5 pMHC Recovery Rate Examples:")
        for i, (epitope, rate) in enumerate(list(pmhc_recovery_rates.items())[:5]):
            gen_logger.info(f"  {epitope}: {rate:.4f}")
        if len(pmhc_recovery_rates) > 5:
            gen_logger.info(f"  ... and {len(pmhc_recovery_rates) - 5} more pMHCs")
    else:
        gen_logger.info("=" * 60)
        gen_logger.info("Sequence Recovery Rate Calculation")
        gen_logger.info("=" * 60)
        gen_logger.info("No true TCR data provided, cannot calculate sequence recovery rate")
    
    if args.generation_mode == "ensemble" and hasattr(args, '_ensemble_method_stats'):
        method_stats_list = args._ensemble_method_stats
        if method_stats_list:
            methods = args.ensemble_methods.split(',')
            avg_method_ratios = {}
            for method in methods:
                avg_ratio = np.mean([stats[method] for stats in method_stats_list])
                avg_method_ratios[method] = avg_ratio
                results_data[f'final_ratio_{method}'] = avg_ratio
            
            gen_logger.info("=" * 60)
            gen_logger.info("Ensemble Final Method Ratio Statistics")
            gen_logger.info("=" * 60)
            gen_logger.info(f"Generation ratios: {args.generation_ratios}")
            if args.mixture_ratios and args.mixture_ratios.strip():
                gen_logger.info(f"Mixture ratios: {args.mixture_ratios}")
            else:
                gen_logger.info(f"Mixture mode: Mix all and sort")
            gen_logger.info(f"Actual average ratios:")
            for method, ratio in avg_method_ratios.items():
                gen_logger.info(f"  {method}: {ratio:.2%}")
    
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    test_name = os.path.splitext(os.path.basename(args.data_path))[0]
    model_save_dir = os.path.dirname(args.model_path)
    experiment_results_path = save_experiment_results_with_metrics(
        model_save_dir, results_data, model_name, test_name, args.generation_mode, df, true_tcr_df
    )
    
    gen_logger.info("=" * 60)
    gen_logger.info(f"Generation completed! Results saved to: {result_path}")
    gen_logger.info(f"Experiment summary saved to: {experiment_results_path}")
    gen_logger.info(f"Log saved to: {log_path}")
    if true_tcr_df is not None:
        gen_logger.info(f"Sequence Recovery Rate: {sequence_recovery_rate:.4f}")
    gen_logger.info("=" * 60)
    
    print(f"Generation completed! Results saved to {result_path}")
    print(f"Experiment results summary saved to {experiment_results_path}")
    print(f"Generation log saved to {log_path}")
    if true_tcr_df is not None:
        print(f"Sequence recovery rate: {sequence_recovery_rate:.4f}")

if __name__=="__main__":
    args = create_parser()
    main(args)