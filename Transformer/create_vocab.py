"""
词汇表构建脚本

用于从预处理后的数据构建词汇表，包括：
1. 分词和统计词频
2. 动态调整词频阈值
3. 生成指定大小的词汇表
"""

import os
import json
import jieba
from collections import Counter
from tqdm import tqdm
import re

def tokenize_text(text, is_chinese=False):
    """
    对文本进行分词
    :param text: 输入文本
    :param is_chinese: 是否为中文文本
    :return: 分词后的token列表
    """
    text = text.strip()
    if is_chinese:
        # 使用jieba进行中文分词
        return list(jieba.cut(text))
    else:
        # 英文分词，保留标点，处理常见缩写
        text = re.sub(r"([?.!,¿])", r" \1 ", text)  # 分离标点
        text = re.sub(r'[" "]+', " ", text)  # 规范化空格
        return text.strip().split()

def build_vocab(file_path, target_size, min_freq=1, is_chinese=False):
    """
    构建词汇表
    :param file_path: 文本文件路径
    :param target_size: 目标词汇表大小
    :param min_freq: 最小词频
    :param is_chinese: 是否为中文
    :return: 词汇表字典
    """
    print(f"\n开始构建词汇表: {file_path}")
    print(f"目标词汇表大小: {target_size}")
    
    # 计算词频
    counter = Counter()
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="计算词频"):
            tokens = tokenize_text(line, is_chinese)
            counter.update(tokens)
    
    print(f"原始词汇量: {len(counter)}")
    
    # 动态调整最小词频以达到目标词汇表大小
    current_size = len([w for w, c in counter.items() if c >= min_freq])
    while current_size > target_size and min_freq < 100:
        min_freq += 1
        current_size = len([w for w, c in counter.items() if c >= min_freq])
    
    print(f"最终最小词频: {min_freq}")
    
    # 创建词汇表，使用与现有词汇表相同的特殊标记顺序
    vocab = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3
    }
    idx = 4
    
    # 添加高频词到词汇表
    for token, freq in counter.most_common():
        if freq >= min_freq and len(vocab) < target_size:
            vocab[token] = idx
            idx += 1
    
    print(f"最终词汇表大小: {len(vocab)}")
    return vocab

def save_vocab(vocab, file_path):
    """保存词汇表到文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"词汇表已保存到: {file_path}")

def main():
    """主函数"""
    # 构建英文词汇表
    print("构建英文词汇表...")
    src_vocab = build_vocab(
        'data/train/en.txt',  # 使用原始训练数据
        target_size=2000,  # 增加词汇表大小
        min_freq=1,  # 降低最小词频到1
        is_chinese=False
    )
    
    print("\n构建中文词汇表...")
    tgt_vocab = build_vocab(
        'data/train/zh.txt',  # 使用原始训练数据
        target_size=4000,  # 中文词汇表可以更大一些
        min_freq=1,  # 降低最小词频到1
        is_chinese=True
    )
    
    # 保存词汇表
    save_vocab(src_vocab, 'data/vocab/src_vocab.json')
    save_vocab(tgt_vocab, 'data/vocab/tgt_vocab.json')

if __name__ == '__main__':
    main() 
 