"""AI 引擎 —— 搜商品 + 拼 Prompt + 调 LLM"""
from typing import Dict, List, Optional

from openai import OpenAI

from . import config
from .product_data import search_products


def _format_products(products: List[Dict]) -> str:
    if not products:
        return "（暂无匹配商品）"
    lines = []
    for p in products:
        lines.append(
            f"[{p['id']}] {p['name']} — ¥{p['price']} "
            f"（{p.get('category', '')} | 库存{p.get('stock', 0)}件）\n"
            f"   {p.get('desc', '')}"
        )
    return "\n".join(lines)


def chat(user_query: str, history: Optional[List[Dict]] = None) -> Dict:
    client = OpenAI(api_key=config.API_KEY, base_url=config.API_BASE)

    products = search_products(user_query, limit=config.MAX_PRODUCTS_IN_CONTEXT)

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-10:])

    user_message = (
        f"【用户问题】{user_query}\n\n"
        f"【商城可推荐的商品】\n{_format_products(products)}"
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
        reply = f"抱歉，AI 服务暂时不可用。请稍后再试。\n（{str(e)}）"

    return {
        "reply": reply,
        "products": [{"id": p["id"], "name": p["name"], "price": p["price"]} for p in products],
    }
