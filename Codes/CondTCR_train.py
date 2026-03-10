import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import WeightedRandomSampler
from collections import Counter, defaultdict
from BERT import BERT, DualTaskBERT, set_seed
from GPT import GPT_Model, get_attn_pad_mask, get_attn_subsequence_mask
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from Data_prepare import make_data_for_seq2seq, make_data_for_gra
from beam_search import BeamHypotheses, expand_inputs
import argparse
from tqdm import tqdm
from accelerate import Accelerator
import logging
import math
import random
from datetime import datetime
# Generation and evaluation modules removed for training-focused script

# Get GPU setting from environment, can be overridden by command line arguments
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "3")

# ==================== Data Augmentation Configuration ====================
AUGMENTATION_CONFIG = {
    # Sample category thresholds
    'ultra_few_threshold': 50,
    'few_shot_threshold': 200,
    
    # Noise parameters (sigma)
    'noise_sigma': {
        'ultra_few': 0.000,
        'few': 0.010,
        'normal': 0.020
    },
    
    # Noise application probability
    'noise_prob': {
        'ultra_few': 0.00,
        'few': 0.10,
        'normal': 0.20
    },
    
    # Conditional dropout probability (p_drop)
    'drop_prob': {
        'ultra_few': {'early': 0.03, 'late': 0.00},
        'few': {'early': 0.03, 'late': 0.00},
        'normal': {'early': 0.20, 'late': 0.00}
    },
    
    # Curriculum learning settings
    'curriculum_switch_ratio': 0.1,
    
    # Balanced sampling settings
    'balanced_sampling': {
        'use_sqrt_weighting': True,
        'max_tail_ratio_in_batch': 0.25,
        'enable_batch_control': True
    },
    
    # CFG inference parameters for generation stage
    'cfg_weights': {
        'ultra_few': 1.7,
        'few': 1.5,
        'normal': 1.4
    },
    
    # Sampling parameters for generation stage
    'sampling_params': {
        'top_p': 0.92,
        'top_k': 60,
        'repetition_penalty': 1.25
    },
    
    # ==================== Data Augmentation Switches ====================
    'enable_augmentation': {
        'balanced_sampling': False,
        'batch_control': False,
        'conditional_noise': False,
        'conditional_dropout': False,
        'curriculum_learning': False
    }
}

# ==================== Generation Mode Description ====================
"""
Supports three generation modes:

1. 'conditional': Pure conditional generation
   - Uses real pMHC condition vectors for decoding
   - Use case: Strictly conditional TCR generation
   - Feature: Highest consistency with training conditions

2. 'unconditional': Pure unconditional generation
   - Uses null condition vectors for decoding
   - Use case: Explore general TCR patterns learned by model
   - Feature: Not constrained by specific pMHC, highest diversity

3. 'cfg': CFG mixed generation (recommended)
   - Combines conditional and unconditional logits: unconditional + weight*(conditional-unconditional)
   - Stratified CFG weights: higher weights for tail samples, lower for head samples
   - Use case: Balance conditional consistency and generation diversity
   - Feature: Adjustable condition strength, avoids collapse and over-diversification
"""

class EarlyStopping:
    """Early stopping mechanism"""
    def __init__(self, patience=20, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def save_checkpoint(self, model):
        """save the best """
        if self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def restore_best_model(self, model):
        """Restore best weights"""
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(model.device if hasattr(model, 'device') else 'cpu') 
                                 for k, v in self.best_weights.items()})

def create_model_save_structure(base_path="../Model_results"):
    """Create model save directory structure"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(base_path, current_time)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir, current_time

def save_training_config_table(model_dir, timestamp, args, augmentation_config, 
                             training_results, best_epoch, stopped_early=False):
    """Save training configuration and results table"""
    config_data = {
        'timestamp': timestamp,
        'model_name': f"model_epoch_{best_epoch}.pth",
        'best_epoch': best_epoch,
        'stopped_early': stopped_early,
        'final_train_loss': training_results['final_train_loss'],
        'best_val_loss': training_results['best_val_loss'],
        'total_epochs_run': training_results['total_epochs_run'],
        
        # Training parameters
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.epoch,
        'random_seed': args.random_seed,
        'freeze_bert': args.freeze_bert,
        'freeze_gpt': args.freeze_gpt,
        
        # Data augmentation configuration
        'balanced_sampling': augmentation_config['enable_augmentation']['balanced_sampling'],
        'conditional_noise': augmentation_config['enable_augmentation']['conditional_noise'],
        'conditional_dropout': augmentation_config['enable_augmentation']['conditional_dropout'],
        'curriculum_learning': augmentation_config['enable_augmentation']['curriculum_learning'],
        'batch_control': augmentation_config['enable_augmentation']['batch_control'],
        
        # Threshold settings
        'ultra_few_threshold': augmentation_config['ultra_few_threshold'],
        'few_shot_threshold': augmentation_config['few_shot_threshold'],
        
        # Sampling parameters
        'top_p': augmentation_config['sampling_params']['top_p'],
        'top_k': augmentation_config['sampling_params']['top_k'],
        'repetition_penalty': augmentation_config['sampling_params']['repetition_penalty'],
        
        # CFG weights
        'cfg_weight_ultra_few': augmentation_config['cfg_weights']['ultra_few'],
        'cfg_weight_few': augmentation_config['cfg_weights']['few'],
        'cfg_weight_normal': augmentation_config['cfg_weights']['normal'],
        
        # Data paths
        'data_path': args.data_path,
        'bert_path': args.bert_path,
        'gpt_path': args.gpt_path,
    }
    
    # Save as CSV file
    config_table_path = os.path.join(model_dir, f'training_config_{timestamp}.csv')
    config_df = pd.DataFrame([config_data])
    config_df.to_csv(config_table_path, index=False)
    
    return config_table_path

def save_detailed_loss_history(model_dir, timestamp, epoch_losses):
    """Save detailed loss history"""
    loss_history_path = os.path.join(model_dir, f'loss_history_{timestamp}.csv')
    loss_df = pd.DataFrame(epoch_losses)
    loss_df.to_csv(loss_history_path, index=False)
    return loss_history_path

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
    """Multi-head attention mechanism"""
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
        V=self.W_V(input_V).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2) #b*heads*seq*d
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
    """Enhanced GRA model with conditional dropout and noise augmentation"""
    def __init__(self, bert, gpt, vocabsize, d_model=768):
        super().__init__()
        self.bert = bert
        self.gpt = gpt
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocabsize, bias=False)
        
        self.null_condition = nn.Parameter(torch.randn(d_model))
        
    def apply_stratified_noise(self, enc_outputs, sample_categories):
        """Apply stratified noise: only add noise to few-shot samples"""
        if not self.training or not AUGMENTATION_CONFIG['enable_augmentation']['conditional_noise']:
            return enc_outputs, torch.zeros(enc_outputs.size(0), dtype=torch.bool, device=enc_outputs.device)
        
        batch_size = enc_outputs.size(0)
        device = enc_outputs.device
        
        noise_decisions = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, category in enumerate(sample_categories):
            if category == 'normal':
                continue
                
            noise_prob = AUGMENTATION_CONFIG['noise_prob'][category]
            if torch.rand(1).item() < noise_prob:
                noise_decisions[i] = True
        
        if noise_decisions.any():
            noise = torch.zeros_like(enc_outputs)
            for i, category in enumerate(sample_categories):
                if noise_decisions[i]:
                    sigma = AUGMENTATION_CONFIG['noise_sigma'][category]
                    noise[i] = torch.randn_like(enc_outputs[i]) * sigma
            
            enc_outputs = enc_outputs + noise
            
        return enc_outputs, noise_decisions
    
    def apply_stratified_dropout(self, enc_outputs, sample_categories, step=None, total_steps=None):
        """Stratified conditional dropout: use different dropout probabilities based on categories"""
        if not self.training or not AUGMENTATION_CONFIG['enable_augmentation']['conditional_dropout']:
            return enc_outputs, torch.zeros(enc_outputs.size(0), dtype=torch.bool, device=enc_outputs.device)
        
        batch_size = enc_outputs.size(0)
        seq_len = enc_outputs.size(1)
        device = enc_outputs.device
        
        stage = 'early'
        if (step is not None and total_steps is not None and 
            AUGMENTATION_CONFIG['enable_augmentation']['curriculum_learning']):
            curriculum_ratio = step / total_steps
            if curriculum_ratio > AUGMENTATION_CONFIG['curriculum_switch_ratio']:
                stage = 'late'
        
        drop_decisions = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, category in enumerate(sample_categories):
            drop_prob = AUGMENTATION_CONFIG['drop_prob'][category][stage]
            if torch.rand(1).item() < drop_prob:
                drop_decisions[i] = True
        
        if drop_decisions.any():
            null_cond = self.null_condition.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            drop_mask = drop_decisions.view(-1, 1, 1)
            enc_outputs = torch.where(drop_mask, null_cond, enc_outputs)
        
        return enc_outputs, drop_decisions
        
    def forward(self, enc_inputs, dec_inputs, masked_pos, sample_categories=None, step=None, total_steps=None):
        enc_outputs_bert, _ = self.bert(enc_inputs, masked_pos)
        
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_outputs_bert)
        
        if sample_categories is not None:
            enc_outputs, noise_applied = self.apply_stratified_noise(enc_outputs, sample_categories)
            
            temp_categories = []
            for i, category in enumerate(sample_categories):
                if noise_applied[i]:
                    temp_categories.append('normal')
                else:
                    temp_categories.append(category)
            
            enc_outputs, drop_applied = self.apply_stratified_dropout(
                enc_outputs, temp_categories, step, total_steps)
        
        dec_outputs_gpt, _ = self.gpt(dec_inputs)
        
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs, dec_outputs_gpt
        )
        
        dec_outputs = dec_outputs + dec_outputs_gpt
        
        dec_logits = self.projection(dec_outputs)
        
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

class StratifiedBalancedDataset:
    """Stratified balanced dataset wrapper"""
    def __init__(self, dataset, config=AUGMENTATION_CONFIG):
        self.dataset = dataset
        self.config = config
        
        self.pmhc_counts = self._count_pmhc_samples()
        
        self.ultra_few_pmhcs = {
            pmhc for pmhc, count in self.pmhc_counts.items() 
            if count <= config['ultra_few_threshold']
        }
        self.few_shot_pmhcs = {
            pmhc for pmhc, count in self.pmhc_counts.items() 
            if config['ultra_few_threshold'] < count <= config['few_shot_threshold']
        }
        
        self.sample_categories = self._categorize_samples()

    def _count_pmhc_samples(self):
        """Count pMHC samples"""
        pmhc_counts = Counter()
        if hasattr(self.dataset, 'data') and isinstance(self.dataset.data, pd.DataFrame) and 'pMHC' in self.dataset.data.columns:
            epitope_list = self.dataset.data['pMHC']
            pmhc_counts.update(epitope_list)
        else:
            for i in range(len(self.dataset)):
                enc_input = self.dataset[i][3] if len(self.dataset[i]) > 3 else self.dataset[i][0]
                pmhc_id = tuple(enc_input.tolist()) if torch.is_tensor(enc_input) else tuple(enc_input)
                pmhc_counts[pmhc_id] += 1
        return pmhc_counts
    
    def _categorize_samples(self):
        """Assign category labels to each sample"""
        categories = []
        for i in range(len(self.dataset)):
            if hasattr(self.dataset, 'get_pmhc_id'):
                pmhc_id = self.dataset.get_pmhc_id(i)
            else:
                if hasattr(self.dataset, 'data') and isinstance(self.dataset.data, pd.DataFrame) and 'pMHC' in self.dataset.data.columns:
                    pmhc_id = self.dataset.data.iloc[i]['pMHC']
                else:
                    enc_input = self.dataset[i][3] if len(self.dataset[i]) > 3 else self.dataset[i][0]
                    pmhc_id = tuple(enc_input.tolist()) if torch.is_tensor(enc_input) else tuple(enc_input)
            
            if pmhc_id in self.ultra_few_pmhcs:
                categories.append('ultra_few')
            elif pmhc_id in self.few_shot_pmhcs:
                categories.append('few')
            else:
                categories.append('normal')
        
        return categories
    
    def log_statistics(self, logger):
        """Write stratification statistics to log file"""
        total_pmhcs = len(self.pmhc_counts)
        ultra_few_count = len(self.ultra_few_pmhcs)
        few_shot_count = len(self.few_shot_pmhcs)
        
        category_stats = Counter(self.sample_categories)
        total_samples = len(self.sample_categories)
        
        logger.info("=" * 50)
        logger.info("Stratification Statistics")
        logger.info("=" * 50)
        logger.info("pMHC Category Distribution:")
        logger.info(f"  Ultra-few (≤{self.config['ultra_few_threshold']}): {ultra_few_count} classes ({ultra_few_count/total_pmhcs:.1%})")
        logger.info(f"  Few-shot ({self.config['ultra_few_threshold']+1}-{self.config['few_shot_threshold']}): {few_shot_count} classes ({few_shot_count/total_pmhcs:.1%})")
        logger.info(f"  Normal (>{self.config['few_shot_threshold']}): {total_pmhcs-ultra_few_count-few_shot_count} classes")
        
        logger.info("Sample Distribution:")
        logger.info(f"  Ultra-few: {category_stats['ultra_few']} samples ({category_stats['ultra_few']/total_samples:.1%})")
        logger.info(f"  Few-shot: {category_stats['few']} samples ({category_stats['few']/total_samples:.1%})")
        logger.info(f"  Normal: {category_stats['normal']} samples ({category_stats['normal']/total_samples:.1%})")
        
    def get_category_statistics(self):
        """Return category statistics dictionary"""
        total_pmhcs = len(self.pmhc_counts)
        ultra_few_count = len(self.ultra_few_pmhcs)
        few_shot_count = len(self.few_shot_pmhcs)
        
        category_stats = Counter(self.sample_categories)
        total_samples = len(self.sample_categories)
        
        return {
            'total_pmhcs': total_pmhcs,
            'ultra_few_pmhcs': ultra_few_count,
            'few_shot_pmhcs': few_shot_count,
            'normal_pmhcs': total_pmhcs - ultra_few_count - few_shot_count,
            'ultra_few_samples': category_stats['ultra_few'],
            'few_shot_samples': category_stats['few'],
            'normal_samples': category_stats['normal'],
            'total_samples': total_samples
        }
    
    def get_sample_category(self, index):
        """Get category of specified sample"""
        return self.sample_categories[index]
    
    def get_balanced_sampler(self, batch_size):
        """Create mild balanced sampler"""
        if not (AUGMENTATION_CONFIG['enable_augmentation']['balanced_sampling'] and 
                self.config['balanced_sampling']['use_sqrt_weighting']):
            return None
        
        weights = []
        for i in range(len(self.dataset)):
            if hasattr(self.dataset, 'get_pmhc_id'):
                pmhc_id = self.dataset.get_pmhc_id(i)
            else:
                if hasattr(self.dataset, 'data') and isinstance(self.dataset.data, pd.DataFrame) and 'pMHC' in self.dataset.data.columns:
                    pmhc_id = self.dataset.data.iloc[i]['pMHC']
                else:
                    enc_input = self.dataset[i][3] if len(self.dataset[i]) > 3 else self.dataset[i][0]
                    pmhc_id = tuple(enc_input.tolist()) if torch.is_tensor(enc_input) else tuple(enc_input)
            
            count = self.pmhc_counts[pmhc_id]
            weight = 1.0 / math.sqrt(count)
            
            if (AUGMENTATION_CONFIG['enable_augmentation']['batch_control'] and 
                self.config['balanced_sampling']['enable_batch_control']):
                category = self.sample_categories[i]
                if category in ['ultra_few', 'few']:
                    weight *= 0.8
                    
            weights.append(weight)
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

def conditional_generation(model, enc_inputs, masked_pos, tcr_idx2token, max_length=32,
                          top_p=None, top_k=None, repetition_penalty=None, device=None):
    """Pure conditional generation: use real pMHC conditions"""
    if top_p is None:
        top_p = AUGMENTATION_CONFIG['sampling_params']['top_p']
    if top_k is None:
        top_k = AUGMENTATION_CONFIG['sampling_params']['top_k'] 
    if repetition_penalty is None:
        repetition_penalty = AUGMENTATION_CONFIG['sampling_params']['repetition_penalty']
    
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
            
            if (next_token == 2).all():  # [SEP] token
                break
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        for i in range(batch_size):
            sequence = decode_tokens_to_sequence(decoder_input[i], tcr_idx2token)
            generated_sequences.append(sequence)
    
    return generated_sequences

"""
Following generation/evaluation functions removed to streamline script:
- unconditional_generation
- cfg_generation_with_stratified_weights
- apply_sampling_constraints
- decode_tokens_to_sequence
- calculate_diversity
- calculate_recovery_rate
- evaluate_on_testset
- and generate branch in main
"""

def cfg_generation_with_stratified_weights(model, enc_inputs, masked_pos, tcr_idx2token, 
                                         sample_categories, max_length=32, beam_size=50, 
                                         top_p=None, top_k=None, repetition_penalty=None, device=None):
    """Stratified CFG mixed generation: use different CFG weights based on sample categories"""
    if top_p is None:
        top_p = AUGMENTATION_CONFIG['sampling_params']['top_p']
    if top_k is None:
        top_k = AUGMENTATION_CONFIG['sampling_params']['top_k'] 
    if repetition_penalty is None:
        repetition_penalty = AUGMENTATION_CONFIG['sampling_params']['repetition_penalty']
    
    model.eval()
    with torch.no_grad():
        batch_size = enc_inputs.size(0)
        
        enc_outputs_bert_cond, _ = model.bert(enc_inputs, masked_pos)
        enc_outputs_cond, _ = model.encoder(enc_inputs, enc_outputs_bert_cond)
        
        null_cond = model.null_condition.unsqueeze(0).unsqueeze(0).expand(
            batch_size, enc_outputs_cond.size(1), -1
        )
        
        decoder_input = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)
        
        generated_sequences = []
        
        for step in range(max_length - 1):
            dec_outputs_gpt, _ = model.gpt(decoder_input)
            
            dec_outputs_cond, _, _ = model.decoder(
                decoder_input, enc_inputs, enc_outputs_cond, dec_outputs_gpt
            )
            dec_outputs_cond = dec_outputs_cond + dec_outputs_gpt
            logits_cond = model.projection(dec_outputs_cond)[:, -1, :]
            
            dec_outputs_uncond, _, _ = model.decoder(
                decoder_input, enc_inputs, null_cond, dec_outputs_gpt
            )
            dec_outputs_uncond = dec_outputs_uncond + dec_outputs_gpt
            logits_uncond = model.projection(dec_outputs_uncond)[:, -1, :]
            
            logits = torch.zeros_like(logits_cond)
            for i in range(batch_size):
                category = sample_categories[i] if i < len(sample_categories) else 'normal'
                cfg_weight = AUGMENTATION_CONFIG['cfg_weights'][category]
                logits[i] = logits_uncond[i] + cfg_weight * (logits_cond[i] - logits_uncond[i])
            
            logits = apply_sampling_constraints(logits, decoder_input, top_k, top_p, repetition_penalty)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            
            if (next_token == 2).all():
                break
                
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        for i in range(batch_size):
            sequence = decode_tokens_to_sequence(decoder_input[i], tcr_idx2token)
            generated_sequences.append(sequence)
    
    return generated_sequences

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

def setup_training_logger(model_path):
    """Setup training logger with timestamp"""
    model_save_dir = os.path.dirname(model_path)
    os.makedirs(model_save_dir, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'training_{current_time}.log'
    logging_save_path = os.path.join(model_save_dir, log_filename)
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(logging_save_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, logging_save_path

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

def log_config_parameters(logger):
    """Log current configuration parameters"""
    logger.info("=" * 60)
    logger.info("Current Configuration Parameters")
    logger.info("=" * 60)
    
    logger.info("Data Augmentation Switches:")
    for aug_name, enabled in AUGMENTATION_CONFIG['enable_augmentation'].items():
        logger.info(f"  {aug_name:20}: {'Enabled' if enabled else 'Disabled'}")
    
    logger.info("\nCategory Thresholds:")
    logger.info(f"  ultra_few_threshold: {AUGMENTATION_CONFIG['ultra_few_threshold']}")
    logger.info(f"  few_shot_threshold: {AUGMENTATION_CONFIG['few_shot_threshold']}")

    logger.info(f"\nModel Parameters Freeze Status:")
    logger.info(f"  freeze_bert: {'Enabled' if args.freeze_bert else 'Disabled'}")
    logger.info(f"  freeze_gpt: {'Enabled' if args.freeze_gpt else 'Disabled'}")
    
    if AUGMENTATION_CONFIG['enable_augmentation']['conditional_noise']:
        logger.info("\nNoise Parameters:")
        for category, sigma in AUGMENTATION_CONFIG['noise_sigma'].items():
            prob = AUGMENTATION_CONFIG['noise_prob'][category]
            logger.info(f"  {category:12}: σ={sigma:5.3f}, prob={prob:5.2f}")
    
    if AUGMENTATION_CONFIG['enable_augmentation']['conditional_dropout']:
        logger.info("\nConditional Dropout Probability:")
        for category, probs in AUGMENTATION_CONFIG['drop_prob'].items():
            logger.info(f"  {category:12}: early={probs['early']:.3f}, late={probs['late']:.3f}")
    
    if AUGMENTATION_CONFIG['enable_augmentation']['curriculum_learning']:
        logger.info(f"\nCurriculum Learning Switch Ratio: {AUGMENTATION_CONFIG['curriculum_switch_ratio']}")
    
    if AUGMENTATION_CONFIG['enable_augmentation']['balanced_sampling']:
        logger.info("\nBalanced Sampling Settings:")
        sampling_config = AUGMENTATION_CONFIG['balanced_sampling']
        logger.info(f"  use_sqrt_weighting: {sampling_config['use_sqrt_weighting']}")
        logger.info(f"  max_tail_ratio_in_batch: {sampling_config['max_tail_ratio_in_batch']}")
        logger.info(f"  enable_batch_control: {sampling_config['enable_batch_control']}")


def calculate_unique_sequence_stats(df):
    """Calculate unique sequence statistics"""
    all_sequences = df['beta'].tolist()
    total_sequences = len(all_sequences)
    unique_all_sequences = set(all_sequences)
    unique_all_count = len(unique_all_sequences)
    unique_all_ratio = unique_all_count / total_sequences if total_sequences > 0 else 0
    
    unique_by_category = set()
    category_unique_stats = {}
    
    for category in ['ultra_few', 'few', 'normal']:
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
    """Write generation results statistics to log"""
    logger.info("=" * 60)
    logger.info("Generation Results Statistics")
    logger.info("=" * 60)
    logger.info(f"Generation Mode: {generation_mode}")
    logger.info(f"Total Sequences: {len(df)}")
    
    category_stats = df['sample_category'].value_counts()
    logger.info("\nStatistics by Sample Category:")
    for category in ['ultra_few', 'few', 'normal']:
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
    logger.info(f"   Unique Ratio: {unique_stats['unique_all_ratio']:.1%}")
    
    logger.info("\n2. Merged Statistics After Deduplication by Category:")
    logger.info(f"   Merged Unique Sequences: {unique_stats['merged_unique_count']}")
    logger.info(f"   Merged Unique Ratio: {unique_stats['merged_unique_ratio']:.1%}")
    
    logger.info("\n3. Uniqueness Within Each Category:")
    for category in ['ultra_few', 'few', 'normal']:
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
    logger.info(f"  Shortest Length: {seq_lengths.min()}")
    logger.info(f"  Longest Length: {seq_lengths.max()}")
    logger.info(f"  Standard Deviation: {seq_lengths.std():.1f}")
    
    logger.info("\nSequence Length by Category:")
    for category in ['ultra_few', 'few', 'normal']:
        category_df = df[df['sample_category'] == category]
        if len(category_df) > 0:
            cat_lengths = category_df['beta'].apply(len)
            logger.info(f"  {category:12}: avg={cat_lengths.mean():.1f}, range=[{cat_lengths.min()}-{cat_lengths.max()}]")
    
    sequence_counts = df['beta'].value_counts()
    most_frequent = sequence_counts.head(10)
    logger.info(f"\nTop 10 Most Frequent Sequences:")
    for i, (seq, count) in enumerate(most_frequent.items(), 1):
        logger.info(f"  {i:2d}. '{seq}' appears {count} times")

def save_loss_to_csv(model_save_dir, epoch_losses):
    """Save loss history to CSV file"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'training_loss_history_{current_time}.csv'
    csv_path = os.path.join(model_save_dir, csv_filename)
    
    loss_df = pd.DataFrame(epoch_losses)
    loss_df.to_csv(csv_path, index=False)
    return csv_path

def save_experiment_results(model_save_dir, results_data, model_name):
    """Save experiment results to CSV file"""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'experiment_results_{current_time}.csv'
    results_path = os.path.join(model_save_dir, results_filename)
    
    results_data['model_name'] = model_name
    results_data['timestamp'] = current_time
    
    results_df = pd.DataFrame([results_data])
    results_df.to_csv(results_path, index=False)
    return results_path

def create_parser():
    parser = argparse.ArgumentParser(description="GRA seq2seq Enhanced with Early Stopping",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path",dest="data_path",type=str,help="The data file in .csv format.",required=True)
    parser.add_argument("--model_path",dest="model_path",type=str, help="the path to save/load model.", required=True)
    parser.add_argument("--bert_path",dest="bert_path",type=str, help="the path to load bert.", required=True)
    parser.add_argument("--gpt_path",dest="gpt_path",type=str, help="the path to load gpt.", required=True)
    parser.add_argument("--result_path",dest="result_path",type=str, default="",help="the path to store result.", required=False)
    parser.add_argument("--mode",dest="mode",type=str, help="train or generate", required=True)
    parser.add_argument("--epoch",dest="epoch",type=int,help="training epoch",default=200, required=False)
    parser.add_argument("--beam",dest="beam",type=int,help="beam_size",default=50, required=False)
    parser.add_argument("--maxlen",dest="maxlen",type=int,help="maxlen of seq",default=32, required=False)
    parser.add_argument("--batch_size",dest="batch_size",type=int,help="batch_size",default=64, required=False)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,default=5e-5, required=False)
    parser.add_argument("--random_seed",type=int, dest="random_seed", default=1, help="seed for reproductbility",required=False)
    parser.add_argument("--tcr_vocab_path",dest="tcr_vocab_path", default="", type=str,help="The vocab file in .csv format.",required=False)
    parser.add_argument("--pep_vocab_path",dest="pep_vocab_path",default="", type=str,help="The vocab file in .csv format.",required=False)
    parser.add_argument("--freeze_bert",dest="freeze_bert",action='store_true')
    parser.add_argument("--freeze_gpt",dest="freeze_gpt",action='store_true')
    
    parser.add_argument("--patience",dest="patience",type=int,default=15,help="Early stopping patience (epochs to wait)")
    parser.add_argument("--min_delta",dest="min_delta",type=float,default=1e-4,help="Minimum change in validation loss to qualify as improvement")
    
    parser.add_argument("--debug",dest="debug",action='store_true',help="Enable debug mode for verbose output")
    
    parser.add_argument("--enable_balanced_sampling",dest="enable_balanced_sampling",action='store_true',help="Enable balanced sampling")
    parser.add_argument("--enable_conditional_noise",dest="enable_conditional_noise",action='store_true',help="Enable conditional noise")
    parser.add_argument("--enable_conditional_dropout",dest="enable_conditional_dropout",action='store_true',help="Enable conditional dropout")
    parser.add_argument("--enable_curriculum_learning",dest="enable_curriculum_learning",action='store_true',help="Enable curriculum learning")
    parser.add_argument("--enable_batch_control",dest="enable_batch_control",action='store_true',help="Enable batch control inside balanced sampling")

    parser.add_argument("--gpu",dest="gpu",type=str,default="0",help="CUDA_VISIBLE_DEVICES to use, e.g., '0' or '1' )")
    
    args = parser.parse_args()
    return args

def main(args):
    # ==================== Update runtime environment and data augmentation config ====================
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    if args.enable_balanced_sampling:
        AUGMENTATION_CONFIG['enable_augmentation']['balanced_sampling'] = True
    if args.enable_conditional_noise:
        AUGMENTATION_CONFIG['enable_augmentation']['conditional_noise'] = True
    if args.enable_conditional_dropout:
        AUGMENTATION_CONFIG['enable_augmentation']['conditional_dropout'] = True
    if args.enable_curriculum_learning:
        AUGMENTATION_CONFIG['enable_augmentation']['curriculum_learning'] = True
    if args.enable_batch_control:
        AUGMENTATION_CONFIG['enable_augmentation']['batch_control'] = True
    
    if args.mode == 'train':
        model_dir, timestamp = create_model_save_structure()
        
        accelerator = Accelerator()
        set_seed(args.random_seed)
        
        dataset_full = pd.read_csv(args.data_path)

        if args.debug:
            n = int(len(dataset_full) * 0.01)
            dataset_debug = dataset_full.iloc[:n].reset_index(drop=True)
            print(f"[DEBUG] Using first 10% of dataset: {n} samples")
            dataset, validate_dataset, vocab_size_tcr, vocab_size_pep = make_data_for_gra(
                dataset_debug, 32, 55, args.mode, epi_split=True, ratio=0.005)
        else:
            dataset, validate_dataset, vocab_size_tcr, vocab_size_pep = make_data_for_gra(
                dataset_full, 32, 55, args.mode, epi_split=True, ratio=0.005)

        balanced_dataset = StratifiedBalancedDataset(dataset, AUGMENTATION_CONFIG)
        
        if AUGMENTATION_CONFIG['enable_augmentation']['balanced_sampling']:
            sampler = balanced_dataset.get_balanced_sampler(args.batch_size)
            
            def collate_with_categories(batch):
                original_batch = torch.utils.data.dataloader.default_collate(batch)
                categories = [balanced_dataset.get_sample_category(i) for i in range(len(batch))]
                return (*original_batch, categories)
            
            data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                                        collate_fn=collate_with_categories)
        else:
            data_loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        validate_dataloader = Data.DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

        bert = BERT(vocab_size_pep)
        gpt = GPT_Model(vocab_size_tcr)
        bert.load_state_dict(torch.load(args.bert_path))
        gpt.load_state_dict(torch.load(args.gpt_path))
        
        for name, param in bert.named_parameters():
            if name == 'linear.weight' or name == 'linear.bias':
                param.requires_grad = False     
        if args.freeze_bert:
            print("Freeze Parameters of BERT")
            for name, param in bert.named_parameters():
                    param.requires_grad = False   
        if args.freeze_gpt:  
            print("Freeze Parameters of GPT")
            for param in gpt.parameters():
                param.requires_grad = False
                
        model = GRA_Enhanced(bert, gpt, vocab_size_tcr)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = AdamW([{"params": [p for p in model.parameters() if p.requires_grad]},],lr=args.learning_rate)
        total_steps = len(data_loader) * args.epoch
        warmup_steps = 0.1 * total_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        model, optimizer, data_loader, validate_dataloader, scheduler = accelerator.prepare(
            model, optimizer, data_loader, validate_dataloader, scheduler
        )

        log_filename = f'training_{timestamp}.log'
        logging_save_path = os.path.join(model_dir, log_filename)
        
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(logging_save_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print("Logger initialized, ready to run")
        print(f"Training log will be saved to: {logging_save_path}")
        print(f"Model will be saved to: {model_dir}")
        
        log_config_parameters(logger)
        balanced_dataset.log_statistics(logger)
        
        early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4, restore_best_weights=True)
        
        global_step = 0
        epoch_losses = []
        best_epoch = 1
        best_val_loss = float('inf')
        stopped_early = False
        
        logger.info("=" * 60)
        logger.info("Start training with early stopping enabled (patience=10)")
        logger.info("=" * 60)
        
        for epoch in range(args.epoch):
            model.train()
            total_loss = 0
            
            for batch_data in tqdm(data_loader, total=len(data_loader)):
                optimizer.zero_grad()
                
                if AUGMENTATION_CONFIG['enable_augmentation']['balanced_sampling']:
                    dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos, sample_categories = batch_data
                else:
                    dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos = batch_data
                    sample_categories = ['normal'] * enc_input_ids.size(0)
                
                logits_lm, _, _, _ = model(enc_input_ids, dec_input_ids, masked_pos, 
                                         sample_categories=sample_categories,
                                         step=global_step, total_steps=total_steps)

                loss_lm = criterion(logits_lm.transpose(1, 2), output_labels) 
                loss_lm = loss_lm * mask_for_loss
                loss_lm = (loss_lm.float()).mean()
                total_loss += loss_lm.item()
                
                accelerator.backward(loss_lm)
                optimizer.step()
                scheduler.step()
                global_step += 1
            
            loss_epoch = total_loss / len(data_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for dec_input_ids, output_labels, mask_for_loss, enc_input_ids, masked_pos in tqdm(validate_dataloader, total=len(validate_dataloader)):
                    logits_lm, _, _, _ = model(enc_input_ids, dec_input_ids, masked_pos)
                    loss_lm = criterion(logits_lm.transpose(1, 2), output_labels) 
                    loss_lm = loss_lm * mask_for_loss
                    loss_lm = (loss_lm.float()).mean()
                    val_loss += loss_lm.item()
            
            val_loss_epoch = val_loss / len(validate_dataloader)
            
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                best_epoch = epoch + 1
            
            epoch_losses.append({
                'epoch': epoch + 1,
                'train_loss': loss_epoch,
                'val_loss': val_loss_epoch
            })
            
            info_train = f"Epoch:{epoch + 1}, Train Loss:{loss_epoch:.4f}"
            info_val = f"Validation Loss:{val_loss_epoch:.4f}"
            
            if early_stopping(val_loss_epoch, accelerator.unwrap_model(model)):
                stopped_early = True
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Best validation loss: {early_stopping.best_loss:.4f}")
                accelerator.print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            accelerator.print(info_train)
            accelerator.print(info_val)
            logger.info(info_train)
            logger.info(info_val)
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = unwrapped_model.state_dict()
            save_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
            accelerator.save(state_dict, save_path)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        
        if early_stopping.best_weights is not None:
            early_stopping.restore_best_model(unwrapped_model)
            logger.info("Restored best model weights")
        
        balance_flag = AUGMENTATION_CONFIG['enable_augmentation'].get('balanced_sampling', False)
        cfg_flag = AUGMENTATION_CONFIG['enable_augmentation'].get('conditional_dropout', False)
        if balance_flag and cfg_flag:
            balance_str = "both"
        elif balance_flag:
            balance_str = "bal"
        elif cfg_flag:
            balance_str = "cfg"
        else:
            balance_str = "init"
        freeze_bert = getattr(args, "freeze_bert", False)
        freeze_gpt = getattr(args, "freeze_gpt", False)
        if freeze_bert and freeze_gpt:
            freeze_str = "allfreeze"
        elif freeze_bert:
            freeze_str = "freezebert"
        elif freeze_gpt:
            freeze_str = "freezegpt"
        else:
            freeze_str = "unfreeze"
        final_model_path = os.path.join(
            model_dir, 
            f"{balance_str}_{freeze_str}_best_model_epoch_{best_epoch}.pth"
        )
        accelerator.save(unwrapped_model.state_dict(), final_model_path)
        
        training_results = {
            'final_train_loss': epoch_losses[-1]['train_loss'] if epoch_losses else 0,
            'best_val_loss': best_val_loss,
            'total_epochs_run': len(epoch_losses)
        }
        
        loss_history_path = save_detailed_loss_history(model_dir, timestamp, epoch_losses)
        
        config_table_path = save_training_config_table(
            model_dir, timestamp, args, AUGMENTATION_CONFIG, 
            training_results, best_epoch, stopped_early
        )
        
        logger.info("=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info(f"Early Stopped: {'Yes' if stopped_early else 'No'}")
        logger.info(f"Best Epoch: {best_epoch}")
        logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
        logger.info(f"Total Training Epochs: {len(epoch_losses)}")
        logger.info(f"Final Model Save Path: {final_model_path}")
        logger.info(f"Config Table Save Path: {config_table_path}")
        logger.info(f"Loss History Save Path: {loss_history_path}")
        logger.info("=" * 60)
        
        print(f"Training completed!")
        print(f"Early stopping: {'Yes' if stopped_early else 'No'}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final model saved to: {final_model_path}")
        print(f"Training config table saved to: {config_table_path}")
        print(f"Loss history saved to: {loss_history_path}")

if __name__=="__main__":
    args = create_parser()
    main(args)