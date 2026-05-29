"""AI 引擎 —— 搜组件 + 拼 Prompt + 调 LLM"""
import sys
import os
from typing import Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from catalog_data import search_entries


def _format_entries(entries: List[Dict]) -> str:
    if not entries:
        return "（暂无匹配组件）"
    lines = []
    for e in entries:
        tags = ", ".join(e["tags"][:6]) if e["tags"] else "无"
        requires = ", ".join(e["requires"][:4]) if e["requires"] else "无"
        provides = ", ".join(e["provides"][:4]) if e["provides"] else "无"
        lines.append(
            f"[{e['key']}] {e['title']} ({e['kind']})\n"
            f"  摘要: {e['summary'] or '无'}\n"
            f"  标签: {tags}\n"
            f"  需要: {requires}  提供: {provides}"
        )
    return "\n".join(lines)


def chat(user_query: str, history: Optional[List[Dict]] = None) -> Dict:
    client = OpenAI(api_key=config.API_KEY, base_url=config.API_BASE)

    # 搜索相关组件
    entries = search_entries(user_query, limit=config.MAX_ENTRIES_IN_CONTEXT)

    # 如果没搜到，按 kind 模糊搜索
    if not entries:
        for kind in ["adapter", "problem", "representation", "bias", "codec", "head", "plugin", "pipeline"]:
            if kind in user_query.lower():
                entries = search_entries(user_query, kind=kind, limit=config.MAX_ENTRIES_IN_CONTEXT)
                break

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])

    user_message = (
        f"【用户需求】{user_query}\n\n"
        f"【候选组件（从 mlblack catalog 中检索）】\n{_format_entries(entries)}"
    )
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"抱歉，AI 服务暂时不可用。\n（{str(e)}）"

    return {
        "reply": reply,
        "entries": [{"key": e["key"], "title": e["title"], "kind": e["kind"]} for e in entries],
    }
