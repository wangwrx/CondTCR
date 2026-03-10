class ScaledDotProductAttention(nn.Module):
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
        V=self.W_V(input_V).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2) #b*heads*seq*d
        attn_mask=attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context=context.transpose(1,2).reshape(batch_size,-1,self.n_heads*self.d_v)
        output=self.fc(context)
        return self.norm(output + residual), attn
    
class PoswiseFeedForwardNet(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs,attn

class DecoderLayer(nn.Module):
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
    def __init__(self, bert, gpt, vocabsize, d_model=768):
        super().__init__()
        self.bert = bert
        self.gpt = gpt
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocabsize, bias=False)
        
        # 添加空条件向量
        self.null_condition = nn.Parameter(torch.randn(d_model))
        
    def apply_stratified_noise(self, enc_outputs, sample_categories):
        """分层添加噪声：仅对少样本类别添加"""
        if not self.training or not AUGMENTATION_CONFIG['enable_augmentation']['conditional_noise']:
            return enc_outputs, torch.zeros(enc_outputs.size(0), dtype=torch.bool, device=enc_outputs.device)
        
        batch_size = enc_outputs.size(0)
        device = enc_outputs.device
        
        # 为每个样本决定是否添加噪声
        noise_decisions = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, category in enumerate(sample_categories):
            if category == 'normal':
                continue  # 正常样本不加噪声
                
            noise_prob = AUGMENTATION_CONFIG['noise_prob'][category]
            if torch.rand(1).item() < noise_prob:
                noise_decisions[i] = True
        
        if noise_decisions.any():
            # 根据类别使用不同的噪声强度
            noise = torch.zeros_like(enc_outputs)
            for i, category in enumerate(sample_categories):
                if noise_decisions[i]:
                    sigma = AUGMENTATION_CONFIG['noise_sigma'][category]
                    noise[i] = torch.randn_like(enc_outputs[i]) * sigma
            
            enc_outputs = enc_outputs + noise
            
        return enc_outputs, noise_decisions
    
    def apply_stratified_dropout(self, enc_outputs, sample_categories, step=None, total_steps=None):
        """分层条件丢弃：根据类别使用不同的丢弃概率"""
        if not self.training or not AUGMENTATION_CONFIG['enable_augmentation']['conditional_dropout']:
            return enc_outputs, torch.zeros(enc_outputs.size(0), dtype=torch.bool, device=enc_outputs.device)
        
        batch_size = enc_outputs.size(0)
        seq_len = enc_outputs.size(1)
        device = enc_outputs.device
        
        # 确定当前是early还是late阶段
        stage = 'early'
        if (step is not None and total_steps is not None and 
            AUGMENTATION_CONFIG['enable_augmentation']['curriculum_learning']):
            curriculum_ratio = step / total_steps
            if curriculum_ratio > AUGMENTATION_CONFIG['curriculum_switch_ratio']:
                stage = 'late'
        
        # 为每个样本决定是否丢弃条件
        drop_decisions = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i, category in enumerate(sample_categories):
            drop_prob = AUGMENTATION_CONFIG['drop_prob'][category][stage]
            if torch.rand(1).item() < drop_prob:
                drop_decisions[i] = True
        
        if drop_decisions.any():
            # 将选中的样本条件替换为空条件向量
            null_cond = self.null_condition.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            drop_mask = drop_decisions.view(-1, 1, 1)  # 广播
            enc_outputs = torch.where(drop_mask, null_cond, enc_outputs)
        
        return enc_outputs, drop_decisions
        
    def forward(self, enc_inputs, dec_inputs, masked_pos, sample_categories=None, step=None, total_steps=None):
        # BERT编码
        enc_outputs_bert, _ = self.bert(enc_inputs, masked_pos)
        
        # Encoder处理
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, enc_outputs_bert)
        
        # 分层数据增强
        if sample_categories is not None:
            # 1. 先尝试添加噪声
            enc_outputs, noise_applied = self.apply_stratified_noise(enc_outputs, sample_categories)
            
            # 2. 再尝试条件丢弃（避免同一样本既加噪又丢弃）
            temp_categories = []
            for i, category in enumerate(sample_categories):
                # 如果已经加了噪声，就不再丢弃条件（二选一策略）
                if noise_applied[i]:
                    temp_categories.append('normal')  # 临时当作正常样本处理
                else:
                    temp_categories.append(category)
            
            enc_outputs, drop_applied = self.apply_stratified_dropout(
                enc_outputs, temp_categories, step, total_steps)
        
        # GPT解码
        dec_outputs_gpt, _ = self.gpt(dec_inputs)
        
        # Decoder处理
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs, dec_outputs_gpt
        )
        
        # 残差连接
        dec_outputs = dec_outputs + dec_outputs_gpt
        
        # 投影到词汇表
        dec_logits = self.projection(dec_outputs)
        
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns