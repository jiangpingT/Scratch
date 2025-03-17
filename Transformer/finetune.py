# -*- coding: utf-8 -*-
"""
Transformer模型微调

本文件包含Transformer模型的微调功能实现，包括：
- 模型加载和参数冻结
- 微调训练循环
- 微调后模型保存

适配Apple Silicon (M4 Pro)芯片优化
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import Transformer, create_masks
from train import TranslationDataset, build_vocab, LabelSmoothing, SimpleLossCompute

# 检查MPS可用性（Apple Silicon优化）
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {device}")


def freeze_params(model, freeze_encoder=True, freeze_decoder_layers=0):
    """冻结模型参数
    
    参数:
        model: Transformer模型
        freeze_encoder: 是否冻结编码器
        freeze_decoder_layers: 冻结解码器的层数
    """
    # 冻结编码器
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    # 冻结部分解码器层
    if freeze_decoder_layers > 0:
        for i in range(freeze_decoder_layers):
            for param in model.decoder.layers[i].parameters():
                param.requires_grad = False
    
    # 打印可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params}")
    print(f"可训练参数数量: {trainable_params}")
    print(f"冻结参数比例: {(total_params - trainable_params) / total_params:.2%}")


def load_model(model_path, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=100, dropout=0.1):
    """加载模型
    
    参数:
        model_path: 模型路径
        src_vocab_size: 源词汇表大小
        tgt_vocab_size: 目标词汇表大小
        d_model: 模型维度
        n_layers: 编码器和解码器层数
        n_heads: 注意力头数
        d_ff: 前馈网络维度
        max_len: 最大序列长度
        dropout: Dropout比率
        
    返回:
        加载的模型
    """
    # 创建模型
    model = Transformer(
        src_vocab_size, 
        tgt_vocab_size, 
        d_model, 
        n_layers, 
        n_heads, 
        d_ff, 
        max_len, 
        dropout
    ).to(device)
    
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def finetune_epoch(model, data_loader, loss_compute):
    model.train()
    total_loss = 0
    
    with tqdm(data_loader, desc='训练', unit='batch') as pbar:
        for i, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(device), tgt.to(device)
            src_mask, tgt_mask = create_masks(src, tgt[:, :-1])
            
            out = model(src, tgt[:, :-1], src_mask, tgt_mask)
            ntokens = (tgt[:, 1:] != 0).sum().item()
            loss = loss_compute(out, tgt[:, 1:], ntokens)
            
            total_loss += loss
            pbar.set_postfix({'loss': f"{loss/ntokens:.4f}"})
    
    return total_loss / len(data_loader)


def finetune(model, train_data, val_data, criterion, optimizer, epochs, save_path):
    """微调模型
    
    参数:
        model: 模型
        train_data: 训练数据加载器
        val_data: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epochs: 训练轮数
        save_path: 模型保存路径
    """
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # 训练
        train_loss = finetune_epoch(
            model, 
            train_data, 
            SimpleLossCompute(model.generator, criterion, optimizer)
        )
        
        # 验证
        model.eval()
        val_loss = 0
        if val_data is not None:
            with torch.no_grad():
                for src, tgt in val_data:
                    src = src.to(device)
                    tgt = tgt.to(device)
                    
                    # 创建掩码
                    src_mask, tgt_mask = create_masks(src, tgt[:, :-1])
                    
                    # 前向传播
                    out = model(src, tgt[:, :-1], src_mask, tgt_mask)
                    
                    # 计算损失
                    ntokens = (tgt[:, 1:] != 0).sum().item()
                    loss = SimpleLossCompute(model.generator, criterion)(out, tgt[:, 1:], ntokens)
                    val_loss += loss
        
        # 打印训练信息
        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f}")
        if val_data is not None:
            val_loss /= len(val_data)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'src_vocab_size': model.src_vocab_size,
                    'tgt_vocab_size': model.tgt_vocab_size,
                    'd_model': model.d_model,
                    'n_layers': model.n_layers,
                    'n_heads': model.n_heads,
                    'd_ff': model.d_ff,
                    'max_len': model.max_len,
                    'dropout': model.dropout
                }, save_path)
                print(f"✨ 最佳模型已保存到 {save_path} (验证损失: {val_loss:.4f})")
        else:
            # 如果没有验证数据，每个epoch都保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, save_path)
            print(f"模型已保存到 {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Transformer模型微调')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--data_dir', type=str, default='data/finetune', help='微调数据目录')
    parser.add_argument('--src_file', type=str, default='src.txt', help='源语言文件')
    parser.add_argument('--tgt_file', type=str, default='tgt.txt', help='目标语言文件')
    parser.add_argument('--save_path', type=str, default='model_finetuned.pth', help='微调后模型保存路径')
    parser.add_argument('--epochs', type=int, default=5, help='微调轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--freeze_encoder', action='store_true', help='是否冻结编码器')
    parser.add_argument('--freeze_decoder_layers', type=int, default=0, help='冻结解码器的层数')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_layers', type=int, default=6, help='编码器和解码器层数')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')
    parser.add_argument('--max_len', type=int, default=100, help='最大序列长度')
    args = parser.parse_args()
    
    # 构建词汇表
    print("构建词汇表...")
    src_vocab = build_vocab(args.data_dir, args.src_file)
    tgt_vocab = build_vocab(args.data_dir, args.tgt_file)
    print(f"源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    # 创建数据集和数据加载器
    print("创建数据集...")
    train_dataset = TranslationDataset(
        args.data_dir, 
        args.src_file, 
        args.tgt_file, 
        src_vocab, 
        tgt_vocab, 
        args.max_len
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    # 创建验证数据集和数据加载器（如果有验证数据）
    val_dataset = None
    val_loader = None
    val_data_dir = os.path.join(os.path.dirname(args.data_dir), 'val')
    if os.path.exists(val_data_dir):
        val_dataset = TranslationDataset(
            val_data_dir, 
            args.src_file, 
            args.tgt_file, 
            src_vocab, 
            tgt_vocab, 
            args.max_len
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2
        )
    
    # 加载预训练模型
    print("加载预训练模型...")
    model = load_model(
        args.model_path, 
        len(src_vocab), 
        len(tgt_vocab), 
        args.d_model, 
        args.n_layers, 
        args.n_heads, 
        args.d_ff, 
        args.max_len, 
        args.dropout
    )
    
    # 冻结部分参数
    print("冻结部分参数...")
    freeze_params(model, args.freeze_encoder, args.freeze_decoder_layers)
    
    # 创建损失函数和优化器
    criterion = LabelSmoothing(len(tgt_vocab), padding_idx=0, smoothing=0.1).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # 微调模型
    print("开始微调...")
    # 加载微调数据集
    ft_dataset = TranslationDataset(data_dir='data/finetune',
    src_file='en_ft.txt',
    tgt_file='zh_ft.txt',
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab)
    
    ft_loader = DataLoader(ft_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(10):
        pretrained_model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(ft_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # 创建mask
            src_mask, tgt_mask = create_masks(src, tgt)
            
            # 前向传播
            outputs = pretrained_model(src, tgt[:, :-1], src_mask, tgt_mask)
            
            # 计算损失
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                          tgt[:, 1:].contiguous().view(-1))
            
            # 反向传播
            finetune_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), 0.5)
            finetune_optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Finetune Epoch: {epoch+1} | Batch: {batch_idx} | Loss: {loss.item():.4f}')
        
        # 保存微调模型
        if epoch % 2 == 0:
            torch.save(pretrained_model.state_dict(), f'transformer_finetuned_epoch_{epoch}.pt')
    
    print("微调完成!")


if __name__ == '__main__':
    main()