import os
import random
import jieba
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import json
import requests
from tqdm import tqdm
import time

# 下载必要的NLTK数据
try:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("成功下载NLTK数据")
except Exception as e:
    print(f"下载NLTK数据失败: {e}")

class DataAugmenter:
    def __init__(self, src_vocab_file, tgt_vocab_file):
        # 加载词汇表
        with open(src_vocab_file, 'r', encoding='utf-8') as f:
            self.src_vocab = json.load(f)
        with open(tgt_vocab_file, 'r', encoding='utf-8') as f:
            self.tgt_vocab = json.load(f)
        
        # 反向映射
        self.src_id2word = {v: k for k, v in self.src_vocab.items()}
        self.tgt_id2word = {v: k for k, v in self.tgt_vocab.items()}
        
        # 加载中文同义词词典
        try:
            with open('data/chinese_synonyms.json', 'r', encoding='utf-8') as f:
                self.synonyms_dict = json.load(f)
            print("成功加载中文同义词词典")
        except Exception as e:
            print(f"加载中文同义词词典失败: {e}")
            self.synonyms_dict = {}
    
    def synonym_replacement(self, text, n=1, lang='en'):
        """同义词替换
        
        参数:
            text: 输入文本
            n: 替换次数
            lang: 语言('en'或'zh')
        """
        if lang == 'en':
            # 简单的分词方法
            words = text.lower().split()
            # 获取内容词（去除停用词和标点）
            content_words = [word for word in words if word.isalnum()]
            
            if not content_words:
                return text
            
            for _ in range(n):
                if not content_words:
                    break
                    
                # 随机选择一个词进行替换
                rand_idx = random.randint(0, len(content_words)-1)
                word = content_words[rand_idx]
                
                # 获取同义词
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            synonyms.append(lemma.name())
                
                if synonyms:
                    # 选择一个同义词替换
                    synonym = random.choice(synonyms)
                    words = [synonym if w == word else w for w in words]
                    content_words.pop(rand_idx)
            
            return ' '.join(words)
            
        else:  # 中文
            words = list(jieba.cut(text))
            content_words = [word for word in words if len(word.strip()) > 0]
            
            if not content_words:
                return text
            
            for _ in range(n):
                if not content_words:
                    break
                
                rand_idx = random.randint(0, len(content_words)-1)
                word = content_words[rand_idx]
                
                # 使用预先准备的同义词词典
                if word in self.synonyms_dict:
                    synonym = random.choice(self.synonyms_dict[word])
                    words = [synonym if w == word else w for w in words]
                    content_words.pop(rand_idx)
            
            return ''.join(words)
    
    def back_translation(self, text, src_lang='en', tgt_lang='zh'):
        """回译
        使用在线翻译API进行回译（这里需要实现具体的API调用）
        """
        # TODO: 实现在线翻译API调用
        # 这里需要替换为实际的API实现
        return text
    
    def random_insertion(self, text, n=1, lang='en'):
        """随机插入同义词
        
        参数:
            text: 输入文本
            n: 插入次数
            lang: 语言('en'或'zh')
        """
        if lang == 'en':
            words = text.lower().split()
            content_words = [word for word in words if word.isalnum()]
            
            if not content_words:
                return text
                
            for _ in range(n):
                if not content_words:
                    break
                    
                # 随机选择一个词获取其同义词
                word = random.choice(content_words)
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word:
                            synonyms.append(lemma.name())
                            
                if synonyms:
                    # 随机选择插入位置
                    synonym = random.choice(synonyms)
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, synonym)
            
            return ' '.join(words)
            
        else:  # 中文
            words = list(jieba.cut(text))
            content_words = [word for word in words if len(word.strip()) > 0]
            
            if not content_words:
                return text
                
            for _ in range(n):
                if not content_words:
                    break
                    
                word = random.choice(content_words)
                if word in self.synonyms_dict:
                    synonym = random.choice(self.synonyms_dict[word])
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, synonym)
            
            return ''.join(words)
            
    def random_swap(self, text, n=1, lang='en'):
        """随机交换文本中的词
        
        参数:
            text: 输入文本
            n: 交换次数
            lang: 语言('en'或'zh')
        """
        if lang == 'en':
            words = text.lower().split()
            if len(words) < 2:
                return text
                
            for _ in range(n):
                if len(words) < 2:
                    break
                    
                # 随机选择两个不同位置进行交换
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            
            return ' '.join(words)
            
        else:  # 中文
            words = list(jieba.cut(text))
            if len(words) < 2:
                return text
                
            for _ in range(n):
                if len(words) < 2:
                    break
                    
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
            
            return ''.join(words)
            
    def random_deletion(self, text, p=0.1, lang='en'):
        """随机删除文本中的词
        
        参数:
            text: 输入文本
            p: 每个词被删除的概率
            lang: 语言('en'或'zh')
        """
        if lang == 'en':
            words = text.lower().split()
            if len(words) == 1:
                return text
                
            # 保留概率为(1-p)的词
            remaining_words = []
            for word in words:
                if random.random() > p:
                    remaining_words.append(word)
                    
            if not remaining_words:
                # 如果所有词都被删除，至少保留一个
                return random.choice(words)
            
            return ' '.join(remaining_words)
            
        else:  # 中文
            words = list(jieba.cut(text))
            if len(words) == 1:
                return text
                
            remaining_words = []
            for word in words:
                if random.random() > p:
                    remaining_words.append(word)
                    
            if not remaining_words:
                return random.choice(words)
            
            return ''.join(remaining_words)
    
    def augment(self, src_text, tgt_text, methods=None):
        """应用多种数据增强方法
        
        参数:
            src_text: 源语言文本
            tgt_text: 目标语言文本
            methods: 使用的增强方法列表
        """
        if methods is None:
            methods = ['synonym', 'swap', 'delete']
        
        augmented_pairs = []
        
        # 原始数据对
        augmented_pairs.append((src_text, tgt_text))
        
        # 同义词替换
        if 'synonym' in methods:
            src_aug = self.synonym_replacement(src_text, n=1, lang='en')
            tgt_aug = self.synonym_replacement(tgt_text, n=1, lang='zh')
            augmented_pairs.append((src_aug, tgt_aug))
        
        # 随机交换
        if 'swap' in methods:
            src_aug = self.random_swap(src_text, n=1, lang='en')
            tgt_aug = self.random_swap(tgt_text, n=1, lang='zh')
            augmented_pairs.append((src_aug, tgt_aug))
        
        # 随机删除
        if 'delete' in methods:
            src_aug = self.random_deletion(src_text, p=0.1, lang='en')
            tgt_aug = self.random_deletion(tgt_text, p=0.1, lang='zh')
            augmented_pairs.append((src_aug, tgt_aug))
        
        # 回译（如果API可用）
        if 'back_translation' in methods:
            src_aug = self.back_translation(src_text, src_lang='en', tgt_lang='zh')
            tgt_aug = self.back_translation(tgt_text, src_lang='zh', tgt_lang='en')
            augmented_pairs.append((src_aug, tgt_aug))
        
        return augmented_pairs

def main():
    # 配置参数
    src_vocab_file = 'data/src_vocab.json'
    tgt_vocab_file = 'data/tgt_vocab.json'
    input_src_file = 'data/train/en.txt'
    input_tgt_file = 'data/train/zh.txt'
    output_src_file = 'data/train/en_augmented.txt'
    output_tgt_file = 'data/train/zh_augmented.txt'
    
    # 初始化数据增强器
    augmenter = DataAugmenter(src_vocab_file, tgt_vocab_file)
    
    # 读取原始数据
    with open(input_src_file, 'r', encoding='utf-8') as f_src, \
         open(input_tgt_file, 'r', encoding='utf-8') as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()
    
    # 数据增强
    augmented_src_lines = []
    augmented_tgt_lines = []
    
    print("正在进行数据增强...")
    for src_text, tgt_text in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
        src_text = src_text.strip()
        tgt_text = tgt_text.strip()
        
        # 应用多种增强方法
        augmented_pairs = augmenter.augment(src_text, tgt_text)
        
        # 收集增强后的数据
        for src_aug, tgt_aug in augmented_pairs:
            augmented_src_lines.append(src_aug + '\n')
            augmented_tgt_lines.append(tgt_aug + '\n')
    
    # 保存增强后的数据
    print(f"\n原始数据对数量: {len(src_lines)}")
    print(f"增强后数据对数量: {len(augmented_src_lines)}")
    
    with open(output_src_file, 'w', encoding='utf-8') as f_src, \
         open(output_tgt_file, 'w', encoding='utf-8') as f_tgt:
        f_src.writelines(augmented_src_lines)
        f_tgt.writelines(augmented_tgt_lines)
    
    print(f"增强后的数据已保存到:")
    print(f"- {output_src_file}")
    print(f"- {output_tgt_file}")

if __name__ == '__main__':
    main() 