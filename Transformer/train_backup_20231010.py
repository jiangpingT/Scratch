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

# 检查MPS可用性（Apple Silicon优化）
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")

class LabelSmoothing(nn.Module):
    """标签平滑"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def make_optimizer(model, d_model, warmup_steps=4000):
    """自定义优化器配置"""
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
    )
    return optimizer, scheduler

# 训练参数配置
def train(model, train_data, val_data, criterion, optimizer, scheduler, epochs, save_dir):
    """训练函数"""
    best_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
                # 训练统计
        epoch_start_time = time.time()
        batch_count = 0
        total_tokens = 0
        
        with tqdm(train_data, desc=f'Epoch {epoch+1}', unit='batch',
                 bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}') as pbar:
            for src, tgt in pbar:
                batch_count += 1
                src = src.to(device)
                tgt = tgt.to(device)
                
                # 创建掩码
                src_mask, tgt_mask = create_masks(src, tgt[:, :-1])
                
                # 前向传播
                outputs = model(src, tgt[:, :-1], src_mask, tgt_mask)
                
                # 计算损失
                optimizer.zero_grad()
                ntokens = (tgt[:, 1:] != 0).sum().item()
                total_tokens += ntokens
                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                
                # 统计指标
                total_loss += loss.item()
                avg_loss = total_loss / (batch_count * ntokens)
                current_lr = optimizer.param_groups[0]['lr']
                
                # 更新进度条
                pbar.set_postfix(ordered_dict={
                    'loss': f'{loss.item()/ntokens:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'tokens/s': f'{total_tokens/(time.time()-epoch_start_time):.0f}',
                    'mem': f'{torch.mps.current_allocated_memory()/1024**2:.1f}MB'
                })
        
        # 添加设备信息日志
        print(f'\n设备使用统计:\n- 当前分配内存: {torch.mps.current_allocated_memory()/1024**2:.1f}MB\n'
              f'- 峰值内存: {torch.mps.driver_allocated_memory()/1024**2:.1f}MB')
        
        # 验证统计
        epoch_time = time.time() - epoch_start_time
        avg_val_loss = 0
        if val_data is not None:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for src, tgt in val_data:
                    src = src.to(device)
                    tgt = tgt.to(device)
                    src_mask, tgt_mask = create_masks(src, tgt[:, :-1])
                    outputs = model(src, tgt[:, :-1], src_mask, tgt_mask)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), tgt[:, 1:].contiguous().view(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_data)
        
        # 增强epoch统计输出
        print(f'[Epoch {epoch+1}] | ' 
              f'Train Loss: {total_loss/len(train_data):.4f} | ' 
              f'Val Loss: {avg_val_loss:.4f} | ' if val_data is not None else ''
              f'Time: {epoch_time:.1f}s | '
              f'LR: {current_lr:.2e} | '
              f'Batch Size: {len(train_data)} | '
              f'Samples: {len(train_data.dataset)}')
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(save_dir, f'model_best_{timestamp}.pth')
            
            # 添加设备信息和训练时间到保存内容
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': total_loss/len(train_data),
                'val_loss': avg_val_loss,
                'training_time': epoch_time,
                'device': str(device)
            }, model_path)
            print(f'✅ 最佳模型保存到 {model_path}\n'
                  f'    训练损失: {total_loss/len(train_data):.4f} | '
                  f'验证损失: {avg_val_loss:.4f} | '
                  f'训练时长: {epoch_time:.1f}秒')


class TranslationDataset(Dataset):
    """翻译数据集
    
    用于加载和预处理翻译数据。
    
    参数:
        data_dir: 数据目录
        src_file: 源语言文件名
        tgt_file: 目标语言文件名
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        max_len: 最大序列长度
    """
    def __init__(self, data_dir, src_file, tgt_file, src_vocab, tgt_vocab, max_len=100):
        self.src_data = []
        self.tgt_data = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
        # 加载数据
        src_path = os.path.join(data_dir, src_file)
        tgt_path = os.path.join(data_dir, tgt_file)
        
        with open(src_path, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
        
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()
        
        # 确保源语言和目标语言数据长度一致
        assert len(src_lines) == len(tgt_lines), "源语言和目标语言数据长度不一致"
        
        # 预处理数据
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_tokens = ['<sos>'] + src_line.strip().split() + ['<eos>']
            tgt_tokens = ['<sos>'] + tgt_line.strip().split() + ['<eos>']
            
            # 序列截断
            src_tokens = src_tokens[:self.max_len] if len(src_tokens) > self.max_len else src_tokens
            tgt_tokens = tgt_tokens[:self.max_len] if len(tgt_tokens) > self.max_len else tgt_tokens
            
            # 转换索引并填充
            src_indices = self._convert_tokens(src_tokens, self.src_vocab)
            tgt_indices = self._convert_tokens(tgt_tokens, self.tgt_vocab)
            
            # 序列填充
            src_indices += [self.src_vocab['<pad>']] * (self.max_len - len(src_indices))
            tgt_indices += [self.tgt_vocab['<pad>']] * (self.max_len - len(tgt_indices))
            
            self.src_data.append(torch.LongTensor(src_indices))
            self.tgt_data.append(torch.LongTensor(tgt_indices))
    
    def _convert_tokens(self, tokens, vocab):
        """统一处理token转换逻辑"""
        indices = []
        for token in tokens:
            if token not in vocab:
                if token.startswith('<') and token.endswith('>'):
                    raise ValueError(f"特殊符号 {token} 未在词汇表中定义")
                indices.append(vocab.get(token, vocab['<unk>']))
            else:
                indices.append(vocab[token])
        return indices
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src = self.src_data[idx]
        tgt = self.tgt_data[idx]
        return src, tgt

# 主函数入口
def main():
    parser = argparse.ArgumentParser(description='Transformer模型训练')
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--src_file', type=str, required=True, help='源语言文件名')
    parser.add_argument('--tgt_file', type=str, required=True, help='目标语言文件名')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='模型保存目录')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--max_seq_len', type=int, default=100, help='最大序列长度')
    args = parser.parse_args()

    # 构建词汇表
    src_vocab = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}
    tgt_vocab = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}

    # 初始化模型
    model = Transformer(
        len(src_vocab),
        len(tgt_vocab),
        d_model=args.d_model,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_seq_len=args.max_seq_len
    ).to(device)

    # 准备数据集
    train_dataset = TranslationDataset(
        args.data_dir,
        args.src_file,
        args.tgt_file,
        src_vocab,
        tgt_vocab
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化优化器和损失函数
    optimizer, scheduler = make_optimizer(model, args.d_model)
    criterion = LabelSmoothing(
        size=len(tgt_vocab),
        padding_idx=0,
        smoothing=0.1
    )

    # 开始训练
    train(model, train_loader, None, criterion, optimizer, scheduler, args.epochs, args.save_dir)

if __name__ == '__main__':
    main()

class LabelSmoothing(nn.Module):
    """标签平滑"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def make_optimizer(model, d_model, warmup_steps=4000):
    """自定义优化器配置"""
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
    )
    return optimizer, scheduler

# 优化器配置
    """标签平滑"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def make_optimizer(model, d_model, warmup_steps=4000):
    """自定义优化器配置"""
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: (d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
    )
    return optimizer, scheduler

# 损失函数
class TranslationDataset(Dataset):
    """翻译数据集
    
    用于加载和预处理翻译数据。
    
    参数:
        data_dir: 数据目录
        src_file: 源语言文件名
        tgt_file: 目标语言文件名
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        max_len: 最大序列长度
    """
    def __init__(self, data_dir, src_file, tgt_file, src_vocab, tgt_vocab, max_len=100):
        self.src_data = []
        self.tgt_data = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
        # 加载数据
        src_path = os.path.join(data_dir, src_file)
        tgt_path = os.path.join(data_dir, tgt_file)
        
        with open(src_path, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
        
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()
        
        # 确保源语言和目标语言数据长度一致
        assert len(src_lines) == len(tgt_lines), "源语言和目标语言数据长度不一致"
        
        # 预处理数据
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_tokens = ['<sos>'] + src_line.strip().split() + ['<eos>']
            tgt_tokens = ['<sos>'] + tgt_line.strip().split() + ['<eos>']
            
            # 序列截断
            src_tokens = src_tokens[:self.max_len] if len(src_tokens) > self.max_len else src_tokens
            tgt_tokens = tgt_tokens[:self.max_len] if len(tgt_tokens) > self.max_len else tgt_tokens
            
            # 转换索引并填充
            src_indices = self._convert_tokens(src_tokens, self.src_vocab)
            tgt_indices = self._convert_tokens(tgt_tokens, self.tgt_vocab)
            
            # 序列填充
            src_indices += [self.src_vocab['<pad>']] * (self.max_len - len(src_indices))
            tgt_indices += [self.tgt_vocab['<pad>']] * (self.max_len - len(tgt_indices))
            
            self.src_data.append(torch.LongTensor(src_indices))
            self.tgt_data.append(torch.LongTensor(tgt_indices))
    
    def _convert_tokens(self, tokens, vocab):
        """统一处理token转换逻辑"""
        indices = []
        for token in tokens:
            if token not in vocab:
                if token.startswith('<') and token.endswith('>'):
                    raise ValueError(f"特殊符号 {token} 未在词汇表中定义")
                indices.append(vocab.get(token, vocab['<unk>']))
            else:
                indices.append(vocab[token])
        return indices
    def __len__(self):
        return len(self.src_data)
    def __getitem__(self, idx):
        src = self.src_data[idx]
        tgt = self.tgt_data[idx]
        return src, tgt



    """标签平滑"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.detach())





class TranslationDataset(Dataset):
    """翻译数据集
    
    用于加载和预处理翻译数据。
    
    参数:
        data_dir: 数据目录
        src_file: 源语言文件名
        tgt_file: 目标语言文件名
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        max_len: 最大序列长度
    """
    def __init__(self, data_dir, src_file, tgt_file, src_vocab, tgt_vocab, max_len=100):
        self.src_data = []
        self.tgt_data = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
        # 加载数据
        src_path = os.path.join(data_dir, src_file)
        tgt_path = os.path.join(data_dir, tgt_file)
        
        with open(src_path, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
        
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()
        
        # 确保源语言和目标语言数据长度一致
        assert len(src_lines) == len(tgt_lines), "源语言和目标语言数据长度不一致"
        
        # 预处理数据
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_tokens = ['<sos>'] + src_line.strip().split() + ['<eos>']
            tgt_tokens = ['<sos>'] + tgt_line.strip().split() + ['<eos>']
            
            # 序列截断
            src_tokens = src_tokens[:self.max_len] if len(src_tokens) > self.max_len else src_tokens
            tgt_tokens = tgt_tokens[:self.max_len] if len(tgt_tokens) > self.max_len else tgt_tokens
            
            # 转换索引并填充
            src_indices = self._convert_tokens(src_tokens, self.src_vocab)
            tgt_indices = self._convert_tokens(tgt_tokens, self.tgt_vocab)
            
            # 序列填充
            src_indices += [self.src_vocab['<pad>']] * (self.max_len - len(src_indices))
            tgt_indices += [self.tgt_vocab['<pad>']] * (self.max_len - len(tgt_indices))
            
            self.src_data.append(torch.LongTensor(src_indices))
            self.tgt_data.append(torch.LongTensor(tgt_indices))
    
    def _convert_tokens(self, tokens, vocab):
        """统一处理token转换逻辑"""
        indices = []
        for token in tokens:
            if token not in vocab:
                if token.startswith('<') and token.endswith('>'):
                    raise ValueError(f"特殊符号 {token} 未在词汇表中定义")
                indices.append(vocab.get(token, vocab['<unk>']))
            else:
                indices.append(vocab[token])
        return indices
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src = self.src_data[idx]
        tgt = self.tgt_data[idx]
        return src, tgt

# 主函数入口
def main():
    parser = argparse.ArgumentParser(description='Transformer模型训练')
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--src_file', type=str, required=True, help='源语言文件名')
    parser.add_argument('--tgt_file', type=str, required=True, help='目标语言文件名')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='模型保存目录')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--max_seq_len', type=int, default=100, help='最大序列长度')
    args = parser.parse_args()

    # 构建词汇表
    src_vocab = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}
    tgt_vocab = {'<pad>':0, '<unk>':1, '<sos>':2, '<eos>':3}

    # 初始化模型
    model = Transformer(
        len(src_vocab),
        len(tgt_vocab),
        d_model=args.d_model,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_seq_len=args.max_seq_len
    ).to(device)

    # 准备数据集
    train_dataset = TranslationDataset(
        args.data_dir,
        args.src_file,
        args.tgt_file,
        src_vocab,
        tgt_vocab
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化优化器和损失函数
    optimizer, scheduler = make_optimizer(model, args.d_model)
    criterion = LabelSmoothing(
        size=len(tgt_vocab),
        padding_idx=0,
        smoothing=0.1
    )

    # 开始训练
    train(model, train_loader, None, criterion, optimizer, scheduler, args.epochs, args.save_dir)

if __name__ == '__main__':
    main()