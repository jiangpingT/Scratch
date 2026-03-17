# Gemini 上下文

## 项目概述

这个 `Scratch` 目录是一个多功能的工作区，包含了多个独立的项目和学习笔记，主要围绕深度学习和 Web 开发。

### 主要项目和目录

1.  **`Transformer/`**
    *   **类型**: Python 深度学习项目
    *   **简介**: 这是一个从零开始实现的 Transformer 模型，用于机器翻译任务。项目代码适配了 Apple Silicon (M-系列芯片)，并提供了完整的训练、推理和微调流程。
    *   **技术栈**: Python, PyTorch
    *   **关键文件**:
        *   `model.py`: Transformer 模型的核心定义。
        *   `train.py`: 模型训练脚本。
        *   `inference.py`: 用于执行翻译任务的推理脚本。
        *   `finetune.py`: 微调已有模型的脚本。
        *   `data/`: 包含训练、测试和微调所需的数据集。
    *   **运行与构建**:
        *   **训练**: `python train.py --data_dir data/train --epochs 10 --batch_size 32`
        *   **推理**: `python inference.py --model_path model.pth --input "你好"`
        *   **微调**: `python finetune.py --model_path model.pth --data_dir data/finetune --epochs 5`
        *   项目还提供了一个 `scripts/run.sh` 脚本来简化在 `mps` 设备上的运行。

2.  **`kobe/`**
    *   **类型**: 静态网站项目
    *   **简介**: 一个为纪念篮球巨星科比·布莱恩特而创建的个人网站。网站设计精美，包含了科比的生平、职业生涯、图片集和名言等内容。
    *   **技术栈**: HTML, CSS, JavaScript
    *   **关键文件**:
        *   `index.html`: 网站主页。
        *   `styles.css`: 网站的样式表。
        *   `script.js`: 实现图片画廊等交互功能的脚本。
    *   **使用**: 直接在浏览器中打开 `index.html` 即可浏览。

3.  **`LLM-learning/`**
    *   **类型**: 技术学习笔记
    *   **简介**: 这个目录包含了作者在学习大语言模型（LLM）过程中的详细笔记和心得。内容从基础概念、工具使用，到实践项目（如手搓 Transformer、制作 Agent）的记录，非常详尽。
    *   **关键文件**:
        *   `from-zero.md`: 核心笔记文件，记录了从零开始学习 LLM 的完整历程、遇到的问题和解决方案。
        *   `step-by-stepT/`: 包含对 `Transformer-from-scratch` 项目的逐行代码学习笔记。

## 开发约定

*   **Python 环境**: 项目中存在 `.venv` 目录，表明使用了 Python 虚拟环境。在运行 `Transformer` 项目前，应先激活虚拟环境。
*   **版本控制**: 项目使用 Git 进行版本控制。`.gitignore` 文件排除了常见的 Python 缓存、IDE 配置文件和模型文件。
*   **代码风格**:
    *   Python 代码（`Transformer` 项目）看起来遵循了标准的 PEP 8 风格，并有详细的中文注释。
    *   Web 代码（`kobe` 项目）结构清晰，CSS 样式和 JavaScript 脚本分离。
*   **依赖管理**: `Transformer` 项目的 `README.md` 中提到了 `requirements.txt`，但该文件未在列表中，可能需要手动创建或检查。

## 总结

该目录是一个活跃的个人学习和实践空间。`Transformer` 项目是一个结构完整、文档清晰的深度学习项目。`kobe` 是一个可以直接浏览的静态网站。`LLM-learning` 提供了宝贵的学习和调试经验。

在与 Gemini 交互时，可以利用这些信息来：
*   直接在 `Transformer` 项目中进行代码修改、训练或推理。
*   对 `kobe` 网站进行内容更新或样式调整。
*   基于 `LLM-learning` 中的笔记，继续探讨或实践相关技术。
