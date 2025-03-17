# -*- coding: utf-8 -*-
"""Transformer模型训练

本文件包含Transformer模型的训练功能实现，包括：
- 数据加载和预处理
- 训练循环
- 优化器和学习率调度
- 模型保存和加载

适配Apple Silicon (M4 Pro)芯片优化
"""

import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import Transformer, create_masks
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from datetime import datetime
import jieba
from collections import Counter
import re
import json
import torch.nn.functional as F
from config import (
    MODEL_CONFIG, 
    TRAINING_CONFIG, 
    DATA_CONFIG, 
    SPECIAL_TOKENS,
    DEVICE,
    SAVE_CONFIG
)

def get_device():
    """
    获取可用的计算设备
    :return: torch.device对象
    """
    return DEVICE

# 检查MPS可用性（Apple Silicon优化）
device = DEVICE
print(f"使用设备: {device}")

class LabelSmoothing(nn.Module):
    """标签平滑"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 减去pad和当前token
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def make_optimizer(model, lr=TRAINING_CONFIG['learning_rate']):
    """创建优化器"""
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

def evaluate(model, val_loader, criterion, device):
    """验证集评估
    
    参数:
        model: 模型实例
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        
    返回:
        验证集平均损失
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc="评估中"):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.contiguous().view(-1, output.size(-1)),
                           tgt_output.contiguous().view(-1))
            
            non_pad_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item()
            total_tokens += non_pad_tokens
    
    return total_loss / total_tokens

def create_pad_mask(seq, pad_idx=0):
    """创建填充掩码
    
    参数:
        seq: 输入序列
        pad_idx: 填充标记的索引
        
    返回:
        掩码张量
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt, pad_idx):
    """创建目标序列掩码（包含填充掩码和后续掩码）"""
    sz = tgt.size(1)
    
    # 创建填充掩码
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 创建后续掩码
    subsequent_mask = torch.triu(
        torch.ones((sz, sz), device=tgt.device),
        diagonal=1
    ).type(torch.bool)
    subsequent_mask = subsequent_mask.unsqueeze(0)
    
    # 组合两种掩码
    return pad_mask & ~subsequent_mask

def train(model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, num_epochs):
    best_loss = float('inf')
    best_model_path = 'checkpoints/best_model.pt'
    
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_dataloader):
            src = src.to(TRAINING_CONFIG['device'])
            tgt = tgt.to(TRAINING_CONFIG['device'])
            
            optimizer.zero_grad()
            output = model(src, tgt[:-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[1:].reshape(-1))
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['clip_grad'])
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_dataloader)}], "
                      f"Loss: {loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_dataloader:
                src = src.to(TRAINING_CONFIG['device'])
                tgt = tgt.to(TRAINING_CONFIG['device'])
                output = model(src, tgt[:-1])
                loss = criterion(output.reshape(-1, output.size(-1)), tgt[1:].reshape(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"保存最佳模型，验证损失: {best_loss:.4f}")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存检查点
        if (epoch + 1) % TRAINING_CONFIG['save_interval'] == 0:
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")

def generate_square_subsequent_mask(sz):
    """生成方形后续掩码
    
    参数:
        sz: 序列长度
        
    返回:
        掩码张量
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def tokenize_text(text, is_chinese=False):
    """
    对文本进行分词
    :param text: 输入文本
    :param is_chinese: 是否为中文文本
    :return: 分词后的token列表
    """
    text = text.strip()
    if is_chinese:
        return list(text)  # 中文按字符切分
    else:
        return text.split()  # 英文按空格切分

def build_vocab(file_path, min_freq=TRAINING_CONFIG['min_freq'], is_chinese=False):
    """
    构建词汇表
    :param file_path: 文本文件路径
    :param min_freq: 最小词频
    :param is_chinese: 是否为中文
    :return: 词汇表字典
    """
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = tokenize_text(line, is_chinese)
            counter.update(tokens)
    
    # 创建词汇表
    vocab = {
        SPECIAL_TOKENS['pad']: 0,
        SPECIAL_TOKENS['unk']: 1,
        SPECIAL_TOKENS['sos']: 2,
        SPECIAL_TOKENS['eos']: 3
    }
    idx = 4
    for token, freq in counter.most_common():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1
    
    return vocab

def save_vocab(vocab, file_path):
    """保存词汇表到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_vocab(file_path):
    """从文件加载词汇表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataloaders(rebuild_vocab=False):
    """准备数据加载器"""
    # 构建或加载词汇表
    if rebuild_vocab or not os.path.exists(DATA_CONFIG['src_vocab_file']) or not os.path.exists(DATA_CONFIG['tgt_vocab_file']):
        print("构建词汇表...")
        src_vocab = build_vocab(DATA_CONFIG['train_src_file'], is_chinese=False)
        tgt_vocab = build_vocab(DATA_CONFIG['train_tgt_file'], is_chinese=True)
        
        print("保存词汇表...")
        save_vocab(src_vocab, DATA_CONFIG['src_vocab_file'])
        save_vocab(tgt_vocab, DATA_CONFIG['tgt_vocab_file'])
    else:
        print("加载已有词汇表...")
        src_vocab = load_vocab(DATA_CONFIG['src_vocab_file'])
        tgt_vocab = load_vocab(DATA_CONFIG['tgt_vocab_file'])
    
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集
    print("\n准备训练数据集...")
    train_dataset = TranslationDataset(
        src_file=DATA_CONFIG['train_src_file'],
        tgt_file=DATA_CONFIG['train_tgt_file'],
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_length=MODEL_CONFIG['max_seq_length']
    )
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=DATA_CONFIG['num_workers'],
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    return train_loader, len(src_vocab), len(tgt_vocab)

class TranslationDataset(Dataset):
    """翻译数据集类"""
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_length=MODEL_CONFIG['max_seq_length']):
        """初始化数据集
        
        参数:
            src_file: 源语言文件路径
            tgt_file: 目标语言文件路径
            src_vocab: 源语言词汇表
            tgt_vocab: 目标语言词汇表
            max_length: 最大序列长度
        """
        # 读取源语言和目标语言文件
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_lines = [line.strip() for line in f]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_lines = [line.strip() for line in f]
            
        assert len(self.src_lines) == len(self.tgt_lines), \
            f"源语言和目标语言文件行数不匹配: {len(self.src_lines)} vs {len(self.tgt_lines)}"
            
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.src_lines)
        
    def __getitem__(self, idx):
        """获取数据集中的一个样本
        
        参数:
            idx: 样本索引
            
        返回:
            src_tensor: 源语言张量
            tgt_tensor: 目标语言张量
        """
        # 获取源语言和目标语言文本
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]
        
        # 分词并转换为索引
        src_tokens = tokenize_text(src_text, is_chinese=False)
        src_indices = [self.src_vocab.get(token, self.src_vocab[SPECIAL_TOKENS['unk']]) for token in src_tokens]
        src_indices = [self.src_vocab[SPECIAL_TOKENS['sos']]] + src_indices + [self.src_vocab[SPECIAL_TOKENS['eos']]]
        
        tgt_tokens = tokenize_text(tgt_text, is_chinese=True)  # 中文字符级分词
        tgt_indices = [self.tgt_vocab.get(token, self.tgt_vocab[SPECIAL_TOKENS['unk']]) for token in tgt_tokens]
        tgt_indices = [self.tgt_vocab[SPECIAL_TOKENS['sos']]] + tgt_indices + [self.tgt_vocab[SPECIAL_TOKENS['eos']]]
        
        # 检查序列长度并发出警告
        if len(src_indices) > self.max_length or len(tgt_indices) > self.max_length:
            print(f"警告：序列长度超过最大长度 {self.max_length}，将被截断。源长度：{len(src_indices)}，目标长度：{len(tgt_indices)}")
        
        # 截断或填充到指定长度
        src_indices = src_indices[:self.max_length]
        tgt_indices = tgt_indices[:self.max_length]
        
        src_indices += [self.src_vocab[SPECIAL_TOKENS['pad']]] * (self.max_length - len(src_indices))
        tgt_indices += [self.tgt_vocab[SPECIAL_TOKENS['pad']]] * (self.max_length - len(tgt_indices))
        
        # 转换为张量
        src_tensor = torch.LongTensor(src_indices)
        tgt_tensor = torch.LongTensor(tgt_indices)
        
        return src_tensor, tgt_tensor

def main(args):
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 确保模型保存目录存在
    os.makedirs(SAVE_CONFIG['model_dir'], exist_ok=True)
    
    # 准备数据
    print("准备数据...")
    train_loader, src_vocab_size, tgt_vocab_size = prepare_dataloaders(rebuild_vocab=args.rebuild_vocab)
    
    # 初始化模型
    print("\n初始化模型...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=MODEL_CONFIG['d_model'],
        n_layers=MODEL_CONFIG['n_layers'],
        n_heads=MODEL_CONFIG['n_heads'],
        d_ff=MODEL_CONFIG['d_ff'],
        dropout=MODEL_CONFIG['dropout'],
        max_seq_length=MODEL_CONFIG['max_seq_length']
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 初始化优化器和损失函数
    optimizer = make_optimizer(model)
    criterion = LabelSmoothing(
        size=tgt_vocab_size,
        padding_idx=0,
        smoothing=TRAINING_CONFIG['label_smoothing']
    ).to(device)
    
    # 如果指定了检查点，加载模型状态
    start_epoch = 0
    if args.checkpoint:
        print(f"\n加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"从epoch {start_epoch} 继续训练")
    
    # 训练模型
    print("\n开始训练...")
    train(model, train_loader, train_loader, optimizer, criterion, optimizer, TRAINING_CONFIG['epochs'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    parser.add_argument('--checkpoint', type=str, help='检查点文件路径')
    parser.add_argument('--rebuild_vocab', action='store_true', help='是否重新构建词汇表')
    args = parser.parse_args()
    main(args)