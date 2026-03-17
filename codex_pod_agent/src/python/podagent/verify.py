from __future__ import annotations

from . import config


def _client():
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


def _call_validator(prompt: str, max_tokens: int = 1500) -> str:
    c = _client()
    temp = 1 if config.ANTHROPIC_THINKING else 0.2
    extra = {}
    if config.ANTHROPIC_THINKING:
        extra["thinking"] = {"type": "enabled", "budget_tokens": config.ANTHROPIC_THINK_TOKENS}
    msg = c.messages.create(
        model=config.ANTHROPIC_MODEL if config.ANTHROPIC_BASE_URL else config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temp,
        messages=[{"role": "user", "content": prompt}],
        **extra,
    )
    return msg.content[0].text if hasattr(msg, "content") else str(msg)


def verify_bullets(bullets: str) -> str:
    prompt = f"""
请作为播客编辑审核下述中文要点列表并修正：
- 不读出 Markdown 符号；不包含#、*、•、_、` 等；
- 每条≤2句，信息密度高，避免重复；
- 覆盖 LLM/Agent/机器人相关的新增要点；
输出同样采用条目列表（- 开头），仅输出修正后的要点。

【待审要点】
{bullets}
"""
    return _call_validator(prompt, max_tokens=1200)


def verify_script(script: str) -> str:
    prompt = f"""
请作为播客总监审校下述中文双人对谈脚本：
- 角色A/B交替清晰；
- 去除#、*、•、_、`等符号；
- 避免口头禅与“嘟”之类无意义语气；
- 保持8-10分钟节奏，段落自然；
输出仅返回修订后的脚本文本。

【待审脚本】
{script}
"""
    return _call_validator(prompt, max_tokens=2500)

