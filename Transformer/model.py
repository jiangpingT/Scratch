# -*- coding: utf-8 -*-
"""
Transformer模型实现

本文件包含Transformer模型的核心组件实现，包括：
- 多头注意力机制 (Multi-Head Attention)
- 前馈神经网络 (Feed Forward Network)
- 位置编码 (Positional Encoding)
- Transformer编码器 (Encoder)
- Transformer解码器 (Decoder)
- 完整Transformer模型

适配Apple Silicon (M4 Pro)芯片优化
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG, DEVICE

# 检查MPS可用性（Apple Silicon优化）
device = DEVICE
print(f"使用设备: {device}")


class PositionalEncoding(nn.Module):
    """位置编码
    
    将序列中token的位置信息编码成向量，使模型能够利用序列的顺序信息。
    使用正弦和余弦函数的组合来生成位置编码。
    
    参数:
        d_model: 模型的维度
        dropout: dropout比率
        max_len: 支持的最大序列长度
    """
    def __init__(self, d_model, dropout=MODEL_CONFIG['dropout'], max_len=MODEL_CONFIG['max_seq_length']):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为非参数张量
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量 [seq_len, batch_size, d_model]
            
        返回:
            添加位置编码后的张量
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制
    
    将输入分割成多个头，每个头独立计算注意力，然后合并结果。
    这允许模型关注不同位置的不同表示子空间信息。
    
    参数:
        d_model: 模型的维度
        n_heads: 注意力头的数量
        dropout: dropout比率
    """
    def __init__(self, d_model=MODEL_CONFIG['d_model'], n_heads=MODEL_CONFIG['n_heads'], 
                 dropout=MODEL_CONFIG['dropout']):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """前向传播
        
        参数:
            q: 查询张量 [batch_size, seq_len_q, d_model]
            k: 键张量 [batch_size, seq_len_k, d_model]
            v: 值张量 [batch_size, seq_len_v, d_model]
            mask: 掩码张量
            
        返回:
            注意力输出和注意力权重
        """
        batch_size = q.size(0)
        
        # 线性变换
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 在softmax之前应用dropout
        scores = self.dropout(scores)
        attn = torch.softmax(scores, dim=-1)
        
        # 输出
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.w_o(out)
        
        return out


class FeedForward(nn.Module):
    """前馈神经网络
    
    由两个线性变换和一个ReLU激活函数组成。
    
    参数:
        d_model: 模型的维度
        d_ff: 前馈网络的维度
        dropout: dropout比率
    """
    def __init__(self, d_model=MODEL_CONFIG['d_model'], d_ff=MODEL_CONFIG['d_ff'], 
                 dropout=MODEL_CONFIG['dropout']):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        """前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        返回:
            前馈网络输出
        """
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class RelativePositionalEncoding(nn.Module):
    """相对位置编码
    
    使用相对位置信息而不是绝对位置,可以更好地处理变长序列。
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建相对位置编码矩阵
        pe = torch.zeros(max_len * 2, d_model)
        position = torch.arange(-max_len, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        # 获取相对位置编码
        pe = self.pe[max_len-seq_len:max_len+seq_len-1]
        return self.dropout(pe)


class LayerDropout(nn.Module):
    """LayerDropout正则化
    
    随机丢弃整个层,可以防止过拟合并提高模型鲁棒性。
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            if torch.rand(1) < self.p:
                return torch.zeros_like(x)
            return x / (1 - self.p)
        return x


class EncoderLayer(nn.Module):
    """Transformer编码器层
    
    包含多头自注意力和前馈网络，以及层归一化和残差连接。
    
    参数:
        d_model: 模型的维度
        n_heads: 注意力头的数量
        d_ff: 前馈网络的维度
        dropout: dropout比率
    """
    def __init__(self, d_model=MODEL_CONFIG['d_model'], n_heads=MODEL_CONFIG['n_heads'], 
                 d_ff=MODEL_CONFIG['d_ff'], dropout=MODEL_CONFIG['dropout']):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_dropout = LayerDropout(p=0.1)  # 添加LayerDropout

    def forward(self, x, mask=None):
        """前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量
            
        返回:
            编码器层输出
        """
        # 应用LayerDropout
        if self.training:
            x = self.layer_dropout(x)
            
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=MODEL_CONFIG['d_model'], n_heads=MODEL_CONFIG['n_heads'], 
                 d_ff=MODEL_CONFIG['d_ff'], dropout=MODEL_CONFIG['dropout']):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Encoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class Decoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=MODEL_CONFIG['d_model'], 
                 n_layers=MODEL_CONFIG['n_layers'], n_heads=MODEL_CONFIG['n_heads'], 
                 d_ff=MODEL_CONFIG['d_ff'], dropout=MODEL_CONFIG['dropout'], 
                 max_seq_length=MODEL_CONFIG['max_seq_length']):
        super(Transformer, self).__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, src, src_mask):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        return enc_output
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        
        dec_output = tgt
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        return dec_output
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return self.generator(dec_output)

    def generate(self, dec_output):
        """生成
        
        参数:
            dec_output: 解码器输出 [batch_size, tgt_seq_len, d_model]
            
        返回:
            输出概率
        """
        return self.generator(dec_output)

    def generate_batch(self, src_batch, max_len=50, temperature=1.0, top_k=50, top_p=0.9):
        """批量生成翻译
        
        参数:
            src_batch: 源序列批次 [batch_size, src_len]
            max_len: 最大生成长度
            temperature: 温度系数
            top_k: top-k采样参数
            top_p: nucleus采样参数
            
        返回:
            生成的序列 [batch_size, tgt_len]
        """
        batch_size = src_batch.size(0)
        device = src_batch.device
        
        # 编码源序列
        src_mask = (src_batch != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        encoder_output = self.encode(src_batch, src_mask)
        
        # 初始化生成
        generated = torch.full((batch_size, 1), 2, device=device)  # <sos> token
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            # 生成tgt_mask
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(device)
            
            # 解码当前序列
            decoder_output = self.decode(generated, encoder_output, src_mask, tgt_mask)
            logits = self.generator(decoder_output[:, -1])
            
            # 应用温度
            logits = logits / temperature
            
            # 应用n-gram重复惩罚
            for i in range(batch_size):
                if not finished[i].item():
                    penalties = self._compute_ngram_repeat_penalty(generated[i].tolist())
                    for token, penalty in penalties.items():
                        logits[i, token] *= penalty
            
            # 应用top-k采样
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))  # 确保top_k不超过词汇表大小
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(1).expand_as(logits)
                logits[logits < min_values] = float('-inf')
            
            # 应用nucleus采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # 计算概率分布
            probs = torch.softmax(logits, dim=-1)
            
            # 选择下一个token
            next_token = torch.multinomial(probs, 1)
            
            # 更新生成的序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否完成
            finished = finished | (next_token.squeeze(-1) == 3)  # <eos> token
            if finished.all().item():
                break
        
        return generated

    def clear_cache(self):
        """清除缓存"""
        self.encoder_cache.clear()
        self.decoder_cache.clear()

def create_masks(src, tgt=None):
    """创建掩码
    
    为源序列和目标序列创建掩码。
    
    参数:
        src: 源序列 [batch_size, src_seq_len]
        tgt: 目标序列 [batch_size, tgt_seq_len]
        
    返回:
        源序列掩码和目标序列掩码
    """
    # 源序列掩码（用于填充）
    src_mask = (src != 0).unsqueeze(-2)
    
    # 如果没有目标序列，只返回源序列掩码
    if tgt is None:
        return src_mask
    
    # 目标序列掩码（用于填充和后续位置）
    tgt_mask = (tgt != 0).unsqueeze(-2)
    size = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, size, size), diagonal=1)).bool()
    if tgt.is_cuda:
        nopeak_mask = nopeak_mask.cuda()
    tgt_mask = tgt_mask & nopeak_mask
    
    return src_mask, tgt_mask

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

class Generator(nn.Module):
    """生成器类，用于将解码器的输出转换为目标词汇表上的概率分布"""
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 [batch_size, seq_len, d_model] 或 [batch_size, d_model]
            
        返回:
            logits: 未归一化的预测分数
        """
        if x.dim() == 2:
            # 处理单个时间步的情况
            return self.proj(x)  # [batch_size, vocab_size]
        else:
            # 处理序列的情况
            return self.proj(x)  # [batch_size, seq_len, vocab_size]