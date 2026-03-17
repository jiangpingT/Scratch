"""
AI 播客 Agent 配置文件

统一管理所有配置参数，包括：
- API 密钥
- 论文源配置
- Claude Agent 配置
- TTS 配置
- 调度配置
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).parent

# 加载 .env 文件
load_dotenv(BASE_DIR / '.env')

# ============= API 配置 =============
# Anthropic Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", None)  # 自定义端点（可选）
CLAUDE_MODEL = "claude-sonnet-4-5"  # 使用端点支持的模型名
CLAUDE_MAX_TOKENS = 4096
CLAUDE_TEMPERATURE = 0.7

# OpenAI API (用于 TTS)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TTS_MODEL = "tts-1"  # 或 tts-1-hd 高质量版本
OPENAI_TTS_VOICE_A = "alloy"  # 主持人 A 的声音
OPENAI_TTS_VOICE_B = "echo"   # 主持人 B 的声音

# 备选：ElevenLabs API (可选，音质更好)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
USE_ELEVENLABS = False  # 设为 True 使用 ElevenLabs

# ============= 论文源配置 =============
PAPER_SOURCES = {
    "arxiv": {
        "enabled": True,
        "base_url": "http://export.arxiv.org/api/query",
        "categories": [
            "cs.AI",  # Artificial Intelligence
            "cs.CL",  # Computation and Language
            "cs.LG",  # Machine Learning
            "cs.RO",  # Robotics
        ],
        "max_results": 10,
        "sort_by": "submittedDate",
        "sort_order": "descending"
    },
    "huggingface": {
        "enabled": True,
        "daily_papers_url": "https://huggingface.co/papers",
        "max_results": 5
    },
    "semantic_scholar": {
        "enabled": False,  # 需要 API key
        "api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
        "base_url": "https://api.semanticscholar.org/graph/v1",
        "fields": "title,abstract,authors,year,citationCount"
    }
}

# 论文过滤配置
PAPER_FILTER = {
    "min_hours_ago": 0,      # 最少发布多久之前（小时）
    "max_hours_ago": 24,     # 最多发布多久之前（小时）
    "keywords": [            # 关键词过滤（可选）
        "large language model",
        "LLM",
        "agent",
        "transformer",
        "reasoning",
        "multimodal"
    ],
    "exclude_keywords": [    # 排除关键词
        "medical",
        "healthcare"
    ]
}

# ============= Claude Agent 配置 =============
CONTENT_ANALYZER_CONFIG = {
    "system_prompt": """你是一位资深的 AI 研究专家和技术评论员。
你的任务是分析最新的 AI 论文，提取其核心创新点、技术突破和潜在影响。
请用简洁、易懂的语言总结，适合播客听众理解。

分析时请关注：
1. 核心创新点是什么？
2. 解决了什么问题？
3. 技术突破在哪里？
4. 对行业的潜在影响
5. 与现有技术的比较

输出格式：JSON，包含 title, summary, innovation, impact, key_points 字段。
""",
    "max_papers": 5,  # 每天分析的论文数量
    "output_language": "chinese"
}

PODCAST_GENERATOR_CONFIG = {
    "system_prompt_host_a": """你是播客主持人张三，一位充满好奇心的科技爱好者。
你的风格是：
- 提出有趣的问题，引导对话
- 用类比和例子帮助听众理解
- 保持轻松愉快的氛围
- 偶尔加入幽默元素

请以自然对话的方式提问和回应。
""",
    "system_prompt_host_b": """你是播客嘉宾李四，一位 AI 领域的技术专家。
你的风格是：
- 深入浅出地解释技术概念
- 提供专业见解和行业分析
- 分享具体案例和应用场景
- 保持专业但不失亲和力

请以自然对话的方式回答和讨论。
""",
    "num_rounds": 6,  # 对话轮数
    "target_duration_minutes": 8,  # 目标时长（分钟）
    "opening_template": "大家好，欢迎收听 AI 前沿日报。我是主持人张三。",
    "closing_template": "好的，今天的分享就到这里。感谢收听，我们明天见！"
}

# ============= TTS 配置 =============
TTS_CONFIG = {
    "speed": 1.0,  # 语速 (0.25 - 4.0)
    "output_format": "mp3",
    "sample_rate": 24000,  # Hz
    "add_pauses": True,  # 在段落间添加停顿
    "pause_duration": 0.5,  # 停顿时长（秒）
}

# ============= 音频处理配置 =============
AUDIO_CONFIG = {
    "add_background_music": True,
    "background_music_file": BASE_DIR / "assets" / "background_music.mp3",
    "music_volume": 0.1,  # 背景音乐音量 (0.0 - 1.0)
    "add_intro": True,
    "intro_file": BASE_DIR / "assets" / "intro.mp3",
    "add_outro": True,
    "outro_file": BASE_DIR / "assets" / "outro.mp3",
    "normalize_audio": True,  # 音频标准化
    "output_bitrate": "192k"
}

# ============= 存储配置 =============
STORAGE_CONFIG = {
    "local_output_dir": BASE_DIR / "output" / "podcasts",
    "database_path": BASE_DIR / "database" / "papers.db",
    "enable_cloud_storage": False,  # 是否启用云存储
    "cloud_provider": "aliyun_oss",  # aliyun_oss, aws_s3, 等
    "cloud_bucket": "",
    "cloud_access_key": os.getenv("CLOUD_ACCESS_KEY", ""),
    "cloud_secret_key": os.getenv("CLOUD_SECRET_KEY", ""),
}

# ============= 调度配置 =============
SCHEDULER_CONFIG = {
    "timezone": "Asia/Shanghai",
    "daily_run_time": "09:00",  # 每天运行时间
    "enable_scheduler": True,
    "run_on_startup": False,  # 启动时立即运行一次
}

# ============= 通知配置 =============
NOTIFICATION_CONFIG = {
    "enable_notification": False,
    "notification_method": "wechat",  # wechat, email, webhook
    "wechat_webhook_url": os.getenv("WECHAT_WEBHOOK_URL", ""),
    "email_to": "",
    "email_from": "",
    "smtp_server": "",
}

# ============= 日志配置 =============
LOGGING_CONFIG = {
    "log_dir": BASE_DIR / "logs",
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file_max_bytes": 10 * 1024 * 1024,  # 10MB
    "log_file_backup_count": 5
}

# ============= 上下文管理配置 =============
CONTEXT_CONFIG = {
    "max_history_papers": 100,  # 保留最多多少篇历史论文
    "dedup_window_days": 7,  # 去重窗口（天）
    "track_topics": True,  # 是否追踪热点话题
    "topic_threshold": 3,  # 话题出现多少次算热点
}

# ============= 开发/测试配置 =============
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
TEST_MODE = os.getenv("TEST_MODE", "False").lower() == "true"

if TEST_MODE:
    # 测试模式下的配置覆盖
    PAPER_FILTER["max_hours_ago"] = 168  # 7天
    CONTENT_ANALYZER_CONFIG["max_papers"] = 2
    PODCAST_GENERATOR_CONFIG["num_rounds"] = 3
    SCHEDULER_CONFIG["run_on_startup"] = True
