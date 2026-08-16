"""Flask API 服务 —— 给前端调用的接口"""
import sys
import os

from flask import Flask, request, jsonify

# 确保能找到同目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_engine import chat as ai_chat
from product_data import search_products as search_pd

app = Flask(__name__)

# CORS: 前端跨域调用
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

conversations = {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "mall-ai-assistant"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default")

    if not query:
        return jsonify({"error": "query 不能为空"}), 400

    history = conversations.get(session_id, [])

    result = ai_chat(user_query=query, history=history)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": result["reply"]})
    conversations[session_id] = history

    return jsonify(result)


@app.route("/products", methods=["GET"])
def list_products():
    query = request.args.get("search", "").strip()
    limit = int(request.args.get("limit", 10))
    products = search_pd(query, limit=limit)
    return jsonify({
        "count": len(products),
        "products": [{"id": p["id"], "name": p["name"], "price": p["price"],
                       "category": p.get("category", ""), "desc": p.get("desc", "")}
                      for p in products]
    })


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "default")
    conversations.pop(session_id, None)
    return jsonify({"status": "ok", "message": f"会话 {session_id} 已重置"})


if __name__ == "__main__":
    print("=" * 50)
    print("  Mall AI Assistant")
    print("  POST /chat      - AI 对话")
    print("  GET  /products  - 商品搜索")
    print("  POST /reset     - 重置会话")
    print("  GET  /health    - 健康检查")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
