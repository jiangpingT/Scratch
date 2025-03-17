import torch
import torch.nn as nn
from model import Transformer
import json

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词汇表以获取正确的大小
    with open('data/src_vocab.json', 'r') as f:
        src_vocab = json.load(f)
    with open('data/tgt_vocab.json', 'r') as f:
        tgt_vocab = json.load(f)
    
    # 创建模型参数
    args = {
        'd_model': 256,
        'n_layers': 3,
        'n_heads': 8,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_seq_length': 100
    }
    
    # 创建模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args['d_model'],
        n_layers=args['n_layers'],
        n_heads=args['n_heads'],
        d_ff=args['d_ff'],
        dropout=args['dropout'],
        max_seq_length=args['max_seq_length']
    ).to(device)
    
    # 创建检查点
    checkpoint = {
        'model': model.state_dict(),
        'args': args,
        'epoch': 0,
        'loss': float('inf')
    }
    
    # 保存模型
    torch.save(checkpoint, 'saved_models/model_best.pth')
    print("测试模型已保存到 saved_models/model_best.pth")

if __name__ == '__main__':
    main() 