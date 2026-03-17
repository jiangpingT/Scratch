from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .. import config
from ..context.store import load_history
from ..analysis.rank import rank_items
from ..analysis.cluster import cluster_items
from ..strategy import load_strategy
from .claude_agent import (
    summarize_items as fallback_summary,
    generate_duo_script as fallback_script,
)


def loop_available() -> bool:
    return config.CLAUDE_USE_LOOP


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


def _tools_spec() -> List[Dict[str, Any]]:
    return [
        {
            "name": "get_history",
            "description": "获取近几天的播客历史提纲（中文要点），用于避免重复与延续脉络。",
            "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": []},
        },
        {
            "name": "get_seen_links",
            "description": "获取历史中已播报过的链接列表，用于去重。",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "fetch_url",
            "description": "抓取指定URL页面正文文本（简要），用于核对摘要或补充关键信息。",
            "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },
        {
            "name": "rank_items",
            "description": "根据关键词、时效性与来源对候选条目进行打分排序，返回 topk。",
            "input_schema": {"type": "object", "properties": {"items": {"type": "array"}, "topk": {"type": "integer"}}, "required": ["items"]},
        },
        {
            "name": "cluster_items",
            "description": "使用Jaccard近似聚类，返回主题簇（每簇为条目列表）。",
            "input_schema": {"type": "object", "properties": {"items": {"type": "array"}, "threshold": {"type": "number"}}, "required": ["items"]},
        },
    ]


def _handle_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_history":
        limit = int(args.get("limit") or config.CONTEXT_HISTORY)
        hs = load_history(limit=limit)
        return {"bullets": [h.bullets for h in hs if h.bullets], "dates": [h.date for h in hs]}
    if name == "get_seen_links":
        hs = load_history(limit=config.DEDUP_HISTORY)
        links: List[str] = []
        for h in hs:
            links.extend(h.links)
        return {"links": links}
    if name == "fetch_url":
        url = args.get("url", "")
        try:
            import requests  # type: ignore
            from ..utils.readability import extract_main_text
        except Exception as e:
            return {"error": f"依赖缺失：{e}"}
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            text = extract_main_text(r.text)
            return {"ok": True, "text": text[:5000]}
        except Exception as e:
            return {"error": str(e)}
    if name == "rank_items":
        items = args.get("items") or []
        strat = load_strategy()
        topk = int(args.get("topk") or strat.topk)
        ranked = rank_items(items, topk=topk)
        return {"ranked": [{"score": s, "item": it} for s, it in ranked]}
    if name == "cluster_items":
        items = args.get("items") or []
        strat = load_strategy()
        thr = float(args.get("threshold") or strat.cluster_threshold)
        clusters = cluster_items(items, threshold=thr)
        return {"clusters": clusters}
    return {"error": f"未知工具: {name}"}


def _run_with_tools(system_prompt: str, user_content: str, max_steps: int = 4) -> str:
    client = _anthropic_client()
    tools = _tools_spec()
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": user_content}
    ]

    for _ in range(max_steps):
        extra = {}
        temp = 0.5
        if config.ANTHROPIC_THINKING:
            extra["thinking"] = {"type": "enabled", "budget_tokens": config.ANTHROPIC_THINK_TOKENS}
            temp = 1
        models = [config.ANTHROPIC_MODEL] if config.ANTHROPIC_BASE_URL else [m for m in [config.ANTHROPIC_MODEL, "claude-3-5-sonnet-20241022"] if m]
        last_err = None
        resp = None
        for m in models:
            try:
                resp = client.messages.create(
                    model=m,
                    max_tokens=2500,
                    temperature=temp,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                    **extra,
                )
                break
            except Exception as e:
                last_err = e
                continue
        if resp is None:
            raise last_err
        blocks = getattr(resp, "content", [])
        tool_uses = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            # 返回文本
            texts = [getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"]
            return "\n".join([t for t in texts if t]) or str(resp)

        # 对每个 tool_use 执行并返回结果
        results: List[Dict[str, Any]] = []
        for tu in tool_uses:
            name = getattr(tu, "name", "")
            tool_id = getattr(tu, "id", "")
            input_args = getattr(tu, "input", {}) or {}
            out = _handle_tool(name, input_args)
            results.append({"type": "tool_result", "tool_use_id": tool_id, "content": json.dumps(out, ensure_ascii=False)})

        messages.append({"role": "assistant", "content": blocks})
        messages.append({"role": "user", "content": results})

    # 超过步数上限，尽力返回最后响应文本
    return "工具循环超限，结果可能不完整。"


def summarize_items_with_loop(items: List[Dict]) -> str:
    if not loop_available():
        return fallback_summary(items)
    # 将候选条目以简化JSON传入用户内容，由模型选择调用工具补充历史与去重
    strat = load_strategy()
    user = (
        "【候选条目JSON】\n" + json.dumps(items[:24], ensure_ascii=False) +
        "\n任务：先可选调用 rank_items/cluster_items/get_history/get_seen_links/fetch_url 以筛选与补充，"
        f"再输出中文要点列表（≤{strat.topk}条，每条≤2句），突出新增价值并避免重复。"
    )
    system = (
        "你是AI研究播客的资深编者。优先准确与信息密度，避免夸张措辞；"
        "如果工具返回为空，也要给出合理的提纲。"
    )
    return _run_with_tools(system, user, max_steps=load_strategy().max_tool_steps_summary)


def generate_duo_script_with_loop(bullets: str, duration_min: int = 8) -> str:
    if not loop_available():
        return fallback_script(bullets, duration_min)
    user = (
        f"请把以下要点改写为2人中文播客对谈脚本（A/B），时长约{duration_min}分钟。" \
        "要求：开场30秒、结尾30秒；每个要点以结论收束；专业但友好；A/B轮流。\n\n" \
        f"【要点】\n{bullets}"
    )
    system = (
        "你是专业的播客撰稿人。参考历史脉络以承接上期主题，但避免重复冗余；需要时可调用 get_history。"
    )
    return _run_with_tools(system, user, max_steps=load_strategy().max_tool_steps_script)
