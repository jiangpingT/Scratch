"""
模型配置文件

统一管理Transformer模型的所有配置参数，包括：
- 模型结构参数
- 训练参数
- 数据处理参数
- 推理参数
"""

import os
import torch

# 环境设置
ENV = os.getenv('ENV', 'prod')  # 默认为生产环境
print(f"当前环境: {ENV}")

# 根据环境选择配置
if ENV == 'test':
    # 测试环境配置
    MODEL_CONFIG = {
        'd_model': 256,        # 减小模型维度
        'n_layers': 3,         # 减少编码器和解码器层数
        'n_heads': 8,          # 保持注意力头数
        'd_ff': 1024,         # 减小前馈网络维度
        'dropout': 0.1,       # Dropout比率
        'max_seq_length': 128  # 最大序列长度
    }
    
    TRAINING_CONFIG = {
        'batch_size': 16,      # 减小批次大小
        'epochs': 2,          # 测试轮数
        'learning_rate': 1e-4, # 学习率
        'min_freq': 1,        # 最小词频
        'label_smoothing': 0.1, # 标签平滑系数
        'clip_grad_norm': 1.0, # 梯度裁剪阈值
        'warmup_steps': 100,   # 减少预热步数
        'save_interval': 1     # 每轮都保存
    }
    
    DATA_CONFIG = {
        'train_src_file': 'data/processed/en_test.txt',  # 测试用英文数据
        'train_tgt_file': 'data/processed/zh_test.txt',  # 测试用中文数据
        'src_vocab_file': 'data/ted/en_vocab.json',      # 英文词汇表
        'tgt_vocab_file': 'data/ted/zh_vocab.json',      # 中文词汇表
        'num_workers': 0,      # 减少工作进程数
        'pin_memory': True,    # 保持固定内存
        'prefetch_factor': 2   # 预取因子
    }
else:
    # 生产环境配置
    MODEL_CONFIG = {
        'd_model': 512,        # 模型维度
        'n_layers': 6,         # 编码器和解码器层数
        'n_heads': 8,          # 注意力头数
        'd_ff': 2048,         # 前馈网络维度
        'dropout': 0.2,       # 增加Dropout比率
        'max_seq_length': 256  # 最大序列长度
    }
    
    TRAINING_CONFIG = {
        'batch_size': 32,      # 批次大小
        'epochs': 100,         # 训练轮数
        'learning_rate': 5e-5, # 降低学习率
        'min_freq': 1,        # 最小词频
        'label_smoothing': 0.1, # 标签平滑系数
        'clip_grad_norm': 0.5, # 减小梯度裁剪阈值
        'warmup_steps': 2000,  # 调整预热步数
        'save_interval': 5     # 保存间隔（轮数）
    }
    
    DATA_CONFIG = {
        'train_src_file': 'data/train/en.txt',  # 英文训练数据
        'train_tgt_file': 'data/train/zh.txt',  # 中文训练数据
        'src_vocab_file': 'data/vocab/src_vocab.json',  # 英文词汇表
        'tgt_vocab_file': 'data/vocab/tgt_vocab.json',  # 中文词汇表
        'num_workers': 4,      # 数据加载器工作进程数
        'pin_memory': True,    # 是否将数据加载到固定内存
        'prefetch_factor': 2   # 预取因子
    }

# 推理参数（通用）
INFERENCE_CONFIG = {
    'beam_size': 5,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.9,
    'length_penalty': 0.6,
    'min_length': 3,
    'max_length': 100,
    'repetition_penalty': 1.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'test_src_file': 'data/test/src.txt',
    'test_tgt_file': 'data/test/tgt.txt',
    'batch_size': 32,
    'num_workers': 4
}

# 特殊标记（通用）
SPECIAL_TOKENS = {
    'pad': '<pad>',
    'unk': '<unk>',
    'sos': '<sos>',
    'eos': '<eos>'
}

# 设备配置
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 模型保存配置
SAVE_CONFIG = {
    'model_dir': 'saved_models',  # 模型保存目录
    'checkpoint_format': f"model_{ENV}_" + "{}.pth"  # 检查点文件名格式
} 