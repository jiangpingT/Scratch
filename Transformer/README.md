# Transformer 模型实现

## 项目介绍

本项目实现了一个完整的Transformer模型，包括模型定义、训练、推理和微调功能。项目适配Apple Silicon芯片（如M4 Pro），并针对MacBook进行了优化。

## 项目结构

```
Transformer/
├── model.py          # 模型定义文件
├── train.py          # 训练功能实现
├── inference.py      # 推理功能实现
├── finetune.py       # 微调功能实现
├── data/             # 数据集目录
│   ├── train/        # 训练数据集
│   ├── test/         # 测试数据集
│   └── finetune/     # 微调数据集
└── todo.md           # 项目进度跟踪
```

## 使用方法

### 环境要求

- Python 3.8+
- PyTorch 2.0+（支持Apple Silicon）
- 其他依赖见requirements.txt

### 训练模型

使用训练脚本启动：
```shell
./scripts/run.sh train mps 64 20 data/train en.txt zh.txt
```

参数说明：
- `train`: 运行模式（训练/微调/推理）
- `mps`: 使用Apple Metal加速（M1/M2芯片）
- `64`: 批次大小
- `20`: 训练轮数
- `data/train`: 训练数据目录路径
- `en.txt`: 源语言文本文件名
- `zh.txt`: 目标语言文本文件名

```bash
python train.py --data_dir data/train --epochs 10 --batch_size 32
```

### 推理

```bash
python inference.py --model_path model.pth --input "你好，请问你是谁？"
```

### 微调模型

```bash
python finetune.py --model_path model.pth --data_dir data/finetune --epochs 5
```

## 注意事项

- 本项目针对Apple M4 Pro芯片进行了优化，可以充分利用MPS加速
- 默认使用小型数据集，适合在本地环境快速训练和测试
- 详细的实现说明请参考各文件中的注释