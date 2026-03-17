from __future__ import annotations

from typing import List, Dict

from .. import config
from ..context.store import load_history


def _anthropic_client():
    try:
        import anthropic  # type: ignore
    except Exception as e:
        raise RuntimeError("anthropic 未安装：pip install anthropic") from e
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("缺少 ANTHROPIC_API_KEY")
    kwargs = {"api_key": config.ANTHROPIC_API_KEY}
    if config.ANTHROPIC_BASE_URL:
        kwargs["base_url"] = config.ANTHROPIC_BASE_URL
    return anthropic.Anthropic(**kwargs)


def summarize_items(items: List[Dict]) -> str:
    client = _anthropic_client()
    content = []
    for it in items:
        content.append(f"- {it['title']} | {it['link']}\n{it['summary'][:500]}")
    from .. import config as cfg
    history = load_history(limit=cfg.CONTEXT_HISTORY)
    history_bullets = "\n".join([h.bullets for h in history if h.bullets])
    prompt = (
        "你是资深AI研究播客编剧。请将下列最新论文/观点要点凝练为中文提纲，"
        "聚焦LLM、AI Agent与机器人方向；避免与历史重复；突出新增价值。"
        "每条不超过2句，保留关键信息与影响。\n\n"
        f"【历史脉络摘录】\n{history_bullets}\n\n"
        f"【当日候选】\n" + "\n\n".join(content)
    )
    extra = {}
    temp = 0.5
    if config.ANTHROPIC_THINKING:
        extra["thinking"] = {"type": "enabled", "budget_tokens": config.ANTHROPIC_THINK_TOKENS}
        temp = 1
    # 如配置了自定义网关，则仅使用用户配置的模型名，避免无效回退
    models = [config.ANTHROPIC_MODEL] if config.ANTHROPIC_BASE_URL else [m for m in [config.ANTHROPIC_MODEL, "claude-3-5-sonnet-20241022"] if m]
    last_err = None
    for m in models:
        try:
            msg = client.messages.create(
                model=m,
                max_tokens=1200,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            return msg.content[0].text if hasattr(msg, "content") else str(msg)
        except Exception as e:
            last_err = e
            continue
    raise last_err


def generate_duo_script(bullets: str, duration_min: int = 8) -> str:
    client = _anthropic_client()
    prompt = f"""
你要把要点改写为2人中文播客对谈脚本，主持人A与主持人B，风格专业、清晰、轻松，不闲聊。
要求：
- 时长约{duration_min}分钟；
- 明确分角色台词，使用“A: … / B: …”格式；
- 开场30秒引入主题，结尾30秒总结与下期预告；
- 每个要点以具体结论或建议收束；
以下是要点：
{bullets}
"""
    extra = {}
    temp = 0.7
    if config.ANTHROPIC_THINKING:
        extra["thinking"] = {"type": "enabled", "budget_tokens": config.ANTHROPIC_THINK_TOKENS}
        temp = 1
    models = [config.ANTHROPIC_MODEL] if config.ANTHROPIC_BASE_URL else [m for m in [config.ANTHROPIC_MODEL, "claude-3-5-sonnet-20241022"] if m]
    last_err = None
    for m in models:
        try:
            msg = client.messages.create(
                model=m,
                max_tokens=3000,
                temperature=temp,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            return msg.content[0].text if hasattr(msg, "content") else str(msg)
        except Exception as e:
            last_err = e
            continue
    raise last_err
