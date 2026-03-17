import os
from pathlib import Path
from datetime import datetime

# 先行加载 .env（若存在），确保后续读取环境变量生效
try:
    from dotenv import load_dotenv  # type: ignore

    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(dotenv_path=cwd_env)
    else:
        load_dotenv()
except Exception:
    p = Path.cwd() / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


# 兼容 ANTHROPIC_AUTH_TOKEN 与 ANTHROPIC_API_KEY
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or ""
ANTHROPIC_BASE_URL = env("ANTHROPIC_BASE_URL", "")
ANTHROPIC_MODEL = env("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
ANTHROPIC_THINKING = env("ANTHROPIC_THINKING", "false").lower() in {"1", "true", "yes"}
ANTHROPIC_THINK_TOKENS = int(env("ANTHROPIC_THINK_TOKENS", "1024") or 1024)
OPENAI_API_KEY = env("OPENAI_API_KEY", "")
OPENAI_BASE_URL = env("OPENAI_BASE_URL", "")
OPENAI_TTS_MODEL = env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
CLAUDE_USE_LOOP = env("CLAUDE_USE_LOOP", "false").lower() in {"1", "true", "yes"}
DRY_RUN = env("DRY_RUN", "false").lower() in {"1", "true", "yes"}

TTS_PROVIDER = env("TTS_PROVIDER", "none").lower()
TTS_VOICE_A = env("TTS_VOICE_A", "alloy")
TTS_VOICE_B = env("TTS_VOICE_B", "verse")

PODCAST_FEED_TITLE = env("PODCAST_FEED_TITLE", "Daily AI Research Duo")
PODCAST_FEED_AUTHOR = env("PODCAST_FEED_AUTHOR", "Agent")
PODCAST_BASE_URL = env("PODCAST_BASE_URL", "")
OUTPUT_DIR = env("OUTPUT_DIR", "dist/episodes")

SCHEDULE_TIME = env("SCHEDULE_TIME", "09:00")
CONTEXT_HISTORY = int(env("CONTEXT_HISTORY", "3") or 3)
DEDUP_HISTORY = int(env("DEDUP_HISTORY", "14") or 14)


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")
def _load_dotenv_if_present() -> None:
    # 优先尝试 python-dotenv
    try:
        from dotenv import load_dotenv  # type: ignore

        # 优先当前工作目录下 .env，其次工程根目录
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(dotenv_path=cwd_env)
        else:
            load_dotenv()
        return
    except Exception:
        pass
    # 兜底：手动解析当前目录下 .env（不覆盖已有环境变量）
    p = Path.cwd() / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv_if_present()
