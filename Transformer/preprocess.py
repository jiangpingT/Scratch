"""
数据预处理脚本

用于清理和格式化训练数据，包括：
1. 移除行号和特殊标记
2. 清理标点符号
3. 规范化空格
4. 保存处理后的数据
"""

import re
import os
from tqdm import tqdm

def clean_text(text, is_chinese=False):
    """清理文本
    
    参数:
        text: 输入文本
        is_chinese: 是否为中文文本
        
    返回:
        清理后的文本
    """
    # 移除行号
    text = re.sub(r'^\d+\s+', '', text)
    
    # 移除括号内的内容
    text = re.sub(r'\([^)]*\)', '', text)
    
    # 清理标点符号，但保留一些基本标点
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff,.!?，。！？\s]', '', text)
    
    # 规范化空格
    text = ' '.join(text.split())
    
    if is_chinese:
        # 移除中文文本中的空格，但保留标点
        text = ''.join(text.split())
    
    return text.strip()

def preprocess_file(input_file, output_file, is_chinese=False):
    """预处理文件
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        is_chinese: 是否为中文文本
    """
    # 计算总行数用于进度条
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    
    print(f"\n处理文件: {input_file}")
    print(f"输出到: {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, total=total_lines, desc="预处理进度"):
            # 清理文本
            cleaned_text = clean_text(line, is_chinese)
            if cleaned_text:  # 只保存非空行
                f_out.write(cleaned_text + '\n')

def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs('data/processed', exist_ok=True)
    
    # 预处理增强的英文数据
    print("处理增强的英文数据...")
    preprocess_file(
        'data/train/en_augmented.txt',
        'data/processed/en_augmented_clean.txt',
        is_chinese=False
    )
    
    # 预处理增强的中文数据
    print("\n处理增强的中文数据...")
    preprocess_file(
        'data/train/zh_augmented.txt',
        'data/processed/zh_augmented_clean.txt',
        is_chinese=True
    )
    
    print("\n数据预处理完成！")

if __name__ == '__main__':
    main() 