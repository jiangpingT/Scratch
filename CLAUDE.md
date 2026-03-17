# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个多项目工作空间，包含以下子项目：

1. **Transformer** - Transformer机器翻译模型实现（英中翻译）
2. **kobe** - 科比·布莱恩特纪念网站（HTML/CSS/JS）
3. **LLM-learning** - 大语言模型学习文档和笔记

## 开发环境

- **Python**: 3.8+（通过虚拟环境 `.venv` 管理）
- **设备**: 适配 Apple Silicon (M4 Pro)，使用 MPS (Metal Performance Shaders) 加速
- **默认语言**: 中文（响应、注释、文档）

## Transformer 项目

### 项目架构

核心组件：
- `model.py` - Transformer核心实现（多头注意力、编码器、解码器、位置编码）
- `train.py` - 训练流程（数据加载、优化器、学习率调度、标签平滑）
- `inference.py` - 推理引擎（集束搜索、文本生成）
- `finetune.py` - 微调功能
- `config.py` - 统一配置管理（支持测试/生产环境切换）

数据管道：
- `preprocess.py` - 数据预处理
- `create_vocab.py` - 词汇表构建
- `data_augmentation.py` - 数据增强

### 常用命令

#### 训练模型
```bash
# 基础训练
python train.py

# 从检查点继续训练
python train.py --checkpoint saved_models/model_prod_50.pth

# 重新构建词汇表
python train.py --rebuild_vocab
```

#### 推理
```bash
python inference.py --model_path saved_models/model_prod_100.pth --input "Hello world"
```

#### 微调
```bash
python finetune.py --model_path saved_models/model_prod_100.pth --data_dir data/finetune --epochs 5
```

#### 环境切换
```bash
# 测试环境（小模型，快速验证）
ENV=test python train.py

# 生产环境（完整模型）
ENV=prod python train.py  # 或省略ENV（默认prod）
```

### 配置说明

配置通过 `config.py` 集中管理，支持环境变量 `ENV` 切换：

**测试环境** (`ENV=test`)：
- 小模型（d_model=256, n_layers=3）
- 小批次（batch_size=16）
- 少轮数（epochs=2）
- 快速验证用

**生产环境** (`ENV=prod`, 默认)：
- 完整模型（d_model=512, n_layers=6）
- 正常批次（batch_size=32）
- 完整训练（epochs=100）

### 数据结构

```
data/
├── train/           # 训练数据
│   ├── en.txt      # 英文（源语言）
│   └── zh.txt      # 中文（目标语言）
├── test/           # 测试数据
│   ├── en.txt
│   └── zh.txt
├── finetune/       # 微调数据
│   ├── en_ft.txt
│   └── zh_ft.txt
└── vocab/          # 词汇表
    ├── src_vocab.json
    └── tgt_vocab.json
```

### 设备优化

- 自动检测 MPS（Apple Silicon）并使用加速
- 通过 `config.DEVICE` 统一管理设备配置
- 支持 CPU 回退

### 模型保存

- 检查点格式：`saved_models/model_{ENV}_epoch_{n}.pth`
- 包含：模型权重、优化器状态、epoch、loss
- 保存间隔：测试环境每轮，生产环境每5轮

### 词汇表

- 英文：空格分词
- 中文：字符级分词
- 特殊标记：`<pad>`, `<unk>`, `<sos>`, `<eos>`
- 索引：0-3 保留给特殊标记，4+ 为词汇

## Kobe 纪念网站

静态网站项目，包含：
- `index.html` - 主页面（响应式布局）
- `gallery.html` - 图片集页面
- `styles.css` - 样式表
- `script.js` - 交互脚本

## LLM-learning 文档

学习笔记和资源：
- `from-zero.md` - LLM 入门学习路径
- `from-zero-4-business.md` - 商业应用学习
- `python.md` - Python 相关笔记
- 推荐资源：张老师课程、赋范空间社区文档

## 开发注意事项

1. **Python 项目**：使用 `.venv` 虚拟环境
2. **设备兼容**：代码优先使用 MPS，不可用时回退到 CPU
3. **中文优先**：注释、文档、输出日志使用中文
4. **配置管理**：修改参数优先通过 `config.py`，避免硬编码
5. **数据文件**：确保源语言和目标语言文件行数对齐
