# AI 播客 Agent 🎙️

自动生成 AI 前沿论文播客的智能 Agent，每天早上 09:00 自动获取最新论文、分析内容、生成双人对话播客。

## ✨ 特性

- 🤖 **Claude Agent SDK**: 使用 Loop 和 Context Management 实现智能对话生成
- 📚 **自动论文获取**: 从 arXiv、Hugging Face 等源自动获取最新 AI 论文
- 🧠 **智能内容分析**: Claude 深度分析论文创新点、影响和应用场景
- 🎭 **双人播客**: 自然流畅的主持人对话（好奇型 + 专家型）
- 🔊 **TTS 语音合成**: OpenAI TTS 生成高质量双声道音频
- 🎵 **音频后期**: 自动添加背景音乐、片头片尾、音量标准化
- ⏰ **定时调度**: APScheduler 实现每日自动运行
- 💾 **上下文管理**: SQLite 数据库管理历史论文和热点话题
- 📱 **手机播放**: 支持 RSS feed 或直接播放

## 🏗️ 架构

```
┌──────────────────┐
│  APScheduler     │  每天 09:00 触发
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│              Main Pipeline                    │
│  ┌────────────────────────────────────────┐  │
│  │ 1. Paper Fetcher (arXiv API)           │  │
│  └────────────┬───────────────────────────┘  │
│               ▼                               │
│  ┌────────────────────────────────────────┐  │
│  │ 2. Context Manager (SQLite + 去重)     │  │
│  └────────────┬───────────────────────────┘  │
│               ▼                               │
│  ┌────────────────────────────────────────┐  │
│  │ 3. Content Analyzer (Claude API)       │  │
│  │    - 分析创新点、影响、关键要点         │  │
│  └────────────┬───────────────────────────┘  │
│               ▼                               │
│  ┌────────────────────────────────────────┐  │
│  │ 4. Podcast Generator (Claude Loop)     │  │
│  │    - 多轮对话生成脚本                   │  │
│  │    - 主持人 A ⇄ 主持人 B                │  │
│  └────────────┬───────────────────────────┘  │
│               ▼                               │
│  ┌────────────────────────────────────────┐  │
│  │ 5. TTS Service (OpenAI TTS)            │  │
│  │    - Voice A (alloy) / Voice B (echo)  │  │
│  └────────────┬───────────────────────────┘  │
│               ▼                               │
│  ┌────────────────────────────────────────┐  │
│  │ 6. Audio Processor (pydub)             │  │
│  │    - 合并音频片段                       │  │
│  │    - 添加背景音乐                       │  │
│  │    - 标准化音量                         │  │
│  └────────────┬───────────────────────────┘  │
└───────────────┼──────────────────────────────┘
                ▼
          📁 podcast_YYYYMMDD.mp3
```

## 📦 项目结构

```
ai-podcast-agent/
├── agents/                      # Agent 模块
│   ├── paper_fetcher.py        # 论文获取 Agent
│   ├── context_manager.py      # 上下文管理器
│   ├── content_analyzer.py     # 内容分析 Agent (Claude)
│   └── podcast_generator.py    # 播客生成 Agent (Claude Loop)
├── services/                    # 服务模块
│   ├── tts_service.py          # TTS 服务
│   └── audio_processor.py      # 音频处理器
├── database/                    # 数据库
│   └── papers.db               # SQLite 数据库
├── output/                      # 输出目录
│   └── podcasts/               # 生成的播客文件
├── logs/                        # 日志文件
├── assets/                      # 音频资源
│   ├── intro.mp3               # 片头
│   ├── outro.mp3               # 片尾
│   └── background_music.mp3    # 背景音乐
├── config.py                    # 配置文件
├── scheduler.py                 # 调度器
├── main.py                      # 主程序入口
├── requirements.txt             # Python 依赖
├── .env.example                # 环境变量示例
└── README.md                    # 本文件
```

## 🚀 快速开始

### 1. 克隆项目

```bash
cd /path/to/your/workspace
# 项目已经在 ai-podcast-agent/ 目录
```

### 2. 安装依赖

```bash
cd ai-podcast-agent

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 ffmpeg（音频处理必需）
# Mac:
brew install ffmpeg

# Ubuntu/Debian:
# sudo apt-get install ffmpeg

# Windows:
# 下载 https://ffmpeg.org/download.html
```

### 3. 配置 API 密钥

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
# 必需：
# - ANTHROPIC_API_KEY
# - OPENAI_API_KEY
```

### 4. 测试运行

```bash
# 单次运行模式（立即生成一期播客）
python main.py --mode once

# 测试模式（使用较小的配置参数）
python main.py --mode once --test
```

### 5. 定时调度运行

```bash
# 定时调度模式（每天 09:00 自动运行）
python main.py --mode schedule

# 或者使用 nohup 在后台运行
nohup python main.py --mode schedule > logs/nohup.log 2>&1 &
```

## ⚙️ 配置说明

主要配置在 `config.py` 中：

### 论文源配置

```python
PAPER_SOURCES = {
    "arxiv": {
        "enabled": True,
        "categories": ["cs.AI", "cs.CL", "cs.LG", "cs.RO"],
        "max_results": 10
    }
}
```

### Claude Agent 配置

```python
CONTENT_ANALYZER_CONFIG = {
    "max_papers": 5,  # 每天分析的论文数量
    "output_language": "chinese"
}

PODCAST_GENERATOR_CONFIG = {
    "num_rounds": 6,  # 对话轮数
    "target_duration_minutes": 8
}
```

### TTS 配置

```python
OPENAI_TTS_VOICE_A = "alloy"  # 主持人 A 的声音
OPENAI_TTS_VOICE_B = "echo"   # 主持人 B 的声音
```

### 调度配置

```python
SCHEDULER_CONFIG = {
    "timezone": "Asia/Shanghai",
    "daily_run_time": "09:00",
    "enable_scheduler": True
}
```

## 🎯 使用场景

### 场景 1: 每日自动播客

```bash
# 部署到服务器，每天自动生成
python main.py --mode schedule
```

### 场景 2: 手动生成特定主题播客

1. 修改 `config.py` 中的 `PAPER_FILTER["keywords"]`
2. 运行 `python main.py --mode once`

### 场景 3: 测试和调试

```bash
# 测试各个组件
python agents/paper_fetcher.py      # 测试论文获取
python agents/content_analyzer.py   # 测试内容分析
python agents/podcast_generator.py  # 测试播客生成
python services/tts_service.py      # 测试 TTS
python services/audio_processor.py  # 测试音频处理
```

## 📱 手机播放方案

### 方案 1: 本地文件 + 云同步

1. 播客生成后自动上传到云盘（配置 `STORAGE_CONFIG`）
2. 手机云盘 App 自动同步
3. 使用播客 App 播放

### 方案 2: RSS Feed（推荐）

1. 将播客文件托管到静态服务器或 OSS
2. 生成 RSS feed XML
3. 手机播客 App 订阅私有 RSS

### 方案 3: iOS 快捷指令

创建快捷指令：
1. 获取最新播客 URL
2. 下载到本地
3. 添加到播放列表

## 🔧 高级功能

### 自定义主持人风格

编辑 `config.py` 中的 `system_prompt_host_a` 和 `system_prompt_host_b`：

```python
PODCAST_GENERATOR_CONFIG = {
    "system_prompt_host_a": """你是主持人小明，
    风格：幽默风趣、善于提问...""",

    "system_prompt_host_b": """你是嘉宾博士，
    风格：专业严谨、深入浅出..."""
}
```

### 添加背景音乐

1. 准备音乐文件（MP3 格式）
2. 放到 `assets/background_music.mp3`
3. 配置音量：`AUDIO_CONFIG["music_volume"] = 0.1`

### 云存储集成

```python
STORAGE_CONFIG = {
    "enable_cloud_storage": True,
    "cloud_provider": "aliyun_oss",  # 或 aws_s3
    "cloud_bucket": "your-bucket-name"
}
```

## 💰 成本估算

基于每天生成一期 8 分钟播客：

| 项目 | 成本 | 说明 |
|-----|------|------|
| Claude API | ~$0.5-1/天 | 分析 5 篇论文 + 生成脚本 |
| OpenAI TTS | ~$0.15/天 | ~1000 字符 |
| 云存储 | ~$1-2/月 | 每天 5MB 音频 |
| **总计** | **~$20-30/月** | |

## 🛠️ 故障排查

### 问题 1: 论文获取失败

```bash
# 检查网络连接
curl https://export.arxiv.org/api/query

# 查看日志
tail -f logs/podcast_agent.log
```

### 问题 2: TTS 生成失败

- 检查 OpenAI API 额度
- 确认 API 密钥正确
- 查看错误日志

### 问题 3: 音频处理失败

```bash
# 确认 ffmpeg 已安装
ffmpeg -version

# 重新安装 pydub
pip install --upgrade pydub
```

## 📈 后续计划

- [ ] 支持更多 TTS 提供商（ElevenLabs、Azure TTS）
- [ ] 支持更多论文源（Semantic Scholar、arXiv RSS）
- [ ] 自动生成 RSS feed
- [ ] Web 界面管理
- [ ] 多语言支持
- [ ] 自定义播客模板
- [ ] 听众反馈分析

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

有问题或建议请提 Issue。

---

**享受你的 AI 播客之旅！** 🎧
