# -*- coding: utf-8 -*-
"""
Transformer模型推理

本文件包含Transformer模型的推理功能实现，包括：
- 模型加载
- 文本生成功能
- 批量推理功能

适配Apple Silicon (M4 Pro)芯片优化
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import Transformer
from train import tokenize_text, generate_square_subsequent_mask
import json
from typing import List, Dict, Optional
from config import (
    MODEL_CONFIG, 
    INFERENCE_CONFIG, 
    DATA_CONFIG, 
    SPECIAL_TOKENS,
    DEVICE,
    SAVE_CONFIG
)

# 检查MPS可用性（Apple Silicon优化）
device = DEVICE
print(f"使用设备: {device}")

class BeamSearchNode:
    """集束搜索节点"""
    def __init__(self, prev_node, word_id, log_prob, length):
        self.prev_node = prev_node
        self.word_id = word_id
        self.log_prob = log_prob
        self.length = length
        self.repeated_tokens = {}  # 记录重复token的次数
        
        # 更新重复token计数并应用惩罚
        if prev_node:
            self.repeated_tokens = prev_node.repeated_tokens.copy()
        if word_id in self.repeated_tokens:
            self.repeated_tokens[word_id] += 1
            # 添加重复惩罚，随着重复次数增加惩罚加重
            repeat_penalty = 0.1 * self.repeated_tokens[word_id]
            self.log_prob -= repeat_penalty
        else:
            self.repeated_tokens[word_id] = 1
        
    def eval(self, alpha=1.0):
        """评估节点得分
        
        参数:
            alpha: 长度惩罚系数
            
        返回:
            节点得分
        """
        # 长度惩罚
        lp = ((5.0 + self.length) ** alpha) / (6.0 ** alpha)
        
        # 重复惩罚
        repeat_penalty = sum(count * 0.1 for count in self.repeated_tokens.values() if count > 1)
        
        return self.log_prob / lp - repeat_penalty

    def get_sequence(self):
        """获取完整序列"""
        sequence = []
        node = self
        while node:
            sequence.append(node.word_id)
            node = node.prev_node
        return sequence[::-1]
        
    def __lt__(self, other):
        """用于排序的比较方法"""
        return self.eval() < other.eval()
        
    def __str__(self):
        """字符串表示"""
        return f"BeamNode(word_id={self.word_id}, log_prob={self.log_prob:.2f}, length={self.length}, score={self.eval():.2f})"

class Translator:
    """翻译器类"""
    def __init__(self, model_path, src_vocab_path=DATA_CONFIG['src_vocab_file'], 
                 tgt_vocab_path=DATA_CONFIG['tgt_vocab_file']):
        """初始化翻译器"""
        self.device = DEVICE
        print(f"使用设备: {self.device}")
        
        # 加载词汇表
        print("加载词汇表...")
        self.load_vocabularies(src_vocab_path, tgt_vocab_path)
        print(f"源语言词汇表大小: {self.src_vocab_size}")
        print(f"目标语言词汇表大小: {self.tgt_vocab_size}")
        
        # 加载模型参数以获取正确的词汇表大小
        checkpoint = torch.load(model_path, map_location=self.device)
        model_state = checkpoint['model_state_dict']
        
        # 从模型状态中获取词汇表大小
        tgt_vocab_size = model_state['tgt_embedding.weight'].shape[0]
        src_vocab_size = model_state['src_embedding.weight'].shape[0]
        print(f"模型词汇表大小 - 源语言: {src_vocab_size}, 目标语言: {tgt_vocab_size}")
        
        # 创建模型
        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=MODEL_CONFIG['d_model'],
            n_layers=MODEL_CONFIG['n_layers'],
            n_heads=MODEL_CONFIG['n_heads'],
            d_ff=MODEL_CONFIG['d_ff'],
            dropout=0.0,  # 推理时不使用dropout
            max_seq_length=MODEL_CONFIG['max_seq_length']
        ).to(self.device)
        
        # 加载模型参数
        self.model.load_state_dict(model_state)
        print(f"模型加载自: {model_path}")
        print(f"检查点信息:")
        print(f"- Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"- Loss: {checkpoint.get('loss', 'unknown'):.4f}")
        print(f"- Perplexity: {checkpoint.get('perplexity', 'unknown'):.2f}")
        
        # 设置为评估模式
        self.model.eval()

    def load_vocabularies(self, src_vocab_path, tgt_vocab_path):
        """加载源语言和目标语言词汇表"""
        with open(src_vocab_path, 'r', encoding='utf-8') as f:
            self.src_vocab = json.load(f)
        with open(tgt_vocab_path, 'r', encoding='utf-8') as f:
            self.tgt_vocab = json.load(f)
        
        # 获取词汇表大小
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
        
        # 创建反向映射
        self.src_idx2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx2word = {v: k for k, v in self.tgt_vocab.items()}

    def preprocess_text(self, text: str) -> torch.Tensor:
        """预处理输入文本"""
        # 分词
        tokens = tokenize_text(text.strip().lower(), is_chinese=False)
        
        # 转换为索引
        indices = [self.src_vocab.get(token, 3) for token in tokens]  # 3是的索引
        indices = [1] + indices + [2]  # 添加<sos>和<eos>标记
        
        # 填充到最大长度
        if len(indices) < MODEL_CONFIG['max_seq_length']:
            indices += [0] * (MODEL_CONFIG['max_seq_length'] - len(indices))  # 0是<pad>的索引
        else:
            indices = indices[:MODEL_CONFIG['max_seq_length']]
        
        return torch.LongTensor(indices).unsqueeze(0).to(self.device)

    def postprocess_text(self, tokens: List[int]) -> str:
        """后处理生成的文本"""
        print(f"处理tokens: {tokens}")  # 添加调试信息
        # 转换为词/字
        words = []
        for idx in tokens:
            word = self.tgt_idx2word.get(idx, SPECIAL_TOKENS['unk'])
            print(f"token {idx} -> word {word}")  # 添加调试信息
            if word in ['<pad>', '<sos>', '<eos>']:  # 使用实际的标记名称
                continue
            words.append(word)
        
        # 对于中文，直接拼接
        result = ''.join(words)
        print(f"最终结果: {result}")  # 添加调试信息
        return result

    def beam_search(self, src_tensor, beam_size=5, max_length=50, length_penalty=0.6):
        """使用束搜索生成翻译"""
        batch_size = src_tensor.size(0)
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        
        # 编码
        memory = self.model.encode(src_tensor, src_mask)
        
        # 初始化
        ys = torch.ones(1, 1).fill_(self.tgt_vocab['<sos>']).long().to(self.device)
        
        # 初始化候选序列
        nodes = [BeamSearchNode(None, self.tgt_vocab['<sos>'], 0, 1)]
        
        for i in range(max_length-1):
            # 获取当前所有候选序列
            candidates = []
            
            # 对每个节点进行扩展
            for node in nodes:
                # 如果已经生成了结束符，则不再扩展
                if node.word_id == self.tgt_vocab['<eos>']:
                    candidates.append(node)
                    continue
                
                # 准备输入序列
                seq = node.get_sequence()
                ys = torch.LongTensor(seq).unsqueeze(0).to(self.device)
                
                # 生成目标掩码
                tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(self.device)
                
                # 解码
                out = self.model.decode(ys, memory, src_mask, tgt_mask)
                
                # 获取最后一个时间步的输出
                out = out[:, -1]  # [1, d_model]
                
                # 生成概率分布
                prob = F.softmax(self.model.generator(out), dim=-1)
                
                # 获取top_k个候选
                log_prob, indices = torch.log(prob[0]).topk(beam_size)
                
                # 创建新的候选节点
                for new_k in range(beam_size):
                    word_id = indices[new_k].item()
                    log_p = log_prob[new_k].item()
                    
                    # 如果生成了未知词，给予较大的惩罚
                    if word_id == self.tgt_vocab['<unk>']:
                        log_p -= 10.0
                    
                    node_score = node.log_prob + log_p
                    
                    # 创建新节点
                    new_node = BeamSearchNode(node, word_id, node_score, node.length + 1)
                    candidates.append(new_node)
            
            # 按得分排序并选择前beam_size个节点
            nodes = sorted(candidates, key=lambda x: x.eval(alpha=length_penalty), reverse=True)[:beam_size]
            
            # 如果所有候选都是结束符，提前结束
            if all(node.word_id == self.tgt_vocab['<eos>'] for node in nodes):
                break
        
        # 选择得分最高的序列
        best_node = max(nodes, key=lambda x: x.eval(alpha=length_penalty))
        return best_node.get_sequence()

    def translate(self, src_text: str, temperature: float = 0.7) -> str:
        """翻译单个文本"""
        with torch.no_grad():
            # 预处理输入文本
            src_tensor = self.preprocess_text(src_text)
            
            # 使用集束搜索生成翻译
            output_indices = self.beam_search(src_tensor)
            
            # 后处理生成的文本
            return self.postprocess_text(output_indices)

    def translate_batch(self, texts: List[str], batch_size: int = 32) -> List[str]:
        """批量翻译文本"""
        translations = []
        for i in tqdm(range(0, len(texts), batch_size), desc="批量翻译"):
            batch_texts = texts[i:i + batch_size]
            # 预处理批次
            batch_tensors = [self.preprocess_text(text) for text in batch_texts]
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # 批量生成翻译
            with torch.no_grad():
                for src_tensor in batch_tensor:
                    output_indices = self.beam_search(src_tensor.unsqueeze(0))
                    translation = self.postprocess_text(output_indices)
                    translations.append(translation)
        
        return translations

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用Transformer模型进行翻译')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--test_src', type=str, default=INFERENCE_CONFIG['test_src_file'], help='测试源语言文件')
    parser.add_argument('--test_tgt', type=str, default=INFERENCE_CONFIG['test_tgt_file'], help='测试目标语言文件')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    args = parser.parse_args()
    
    # 创建翻译器
    translator = Translator(
        model_path=args.model_path,
        src_vocab_path=DATA_CONFIG['src_vocab_file'],
        tgt_vocab_path=DATA_CONFIG['tgt_vocab_file']
    )
    
    # 读取测试文件
    with open(args.test_src, 'r', encoding='utf-8') as f:
        test_src = [line.strip() for line in f]
    with open(args.test_tgt, 'r', encoding='utf-8') as f:
        test_tgt = [line.strip() for line in f]
    
    # 进行翻译
    print("\n开始翻译...")
    translations = translator.translate_batch(test_src, args.batch_size)
    
    # 输出结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for src, trans, ref in zip(test_src, translations, test_tgt):
                f.write(f"源文本: {src}\n")
                f.write(f"译文: {trans}\n")
                f.write(f"参考: {ref}\n\n")
        print(f"翻译结果已保存到: {args.output_file}")
    else:
        print("\n翻译结果:")
        for src, trans, ref in zip(test_src, translations, test_tgt):
            print(f"源文本: {src}")
            print(f"译文: {trans}")
            print(f"参考: {ref}\n")

if __name__ == '__main__':
    main()