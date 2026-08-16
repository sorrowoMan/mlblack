"""Flask API —— 给前端/IDE 插件调用"""
import sys
import os

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_engine import chat as ai_chat
from catalog_data import search_entries, get_kinds

app = Flask(__name__)


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    response = jsonify({"error": f"internal server error: {type(exc).__name__}", "detail": str(exc)})
    response.status_code = 500
    return response

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

conversations = {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "catalog-assistant"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "") or "").strip()
    session_id = str(data.get("session_id", "default") or "default").strip() or "default"

    if not query:
        return jsonify({"error": "query 不能为空"}), 400

    history = conversations.get(session_id, [])
    try:
        result = ai_chat(user_query=query, history=history)
        reply = str(result.get("reply", "") or "").strip()
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": reply})
        conversations[session_id] = history
        return jsonify(result)
    except Exception as exc:
        conversations[session_id] = history
        return jsonify({"reply": f"AI 服务响应异常：{type(exc).__name__}: {exc}", "entries": []}), 500


@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q", "").strip()
    kind = request.args.get("kind") or None
    limit = int(request.args.get("limit", 10))
    entries = search_entries(q, kind=kind, limit=limit)
    return jsonify({"count": len(entries), "entries": entries})


@app.route("/kinds", methods=["GET"])
def kinds():
    return jsonify({"kinds": get_kinds()})


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "default")
    conversations.pop(session_id, None)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("=" * 50)
    print("  Catalog AI Assistant")
    print("  POST /chat     - AI 对话")
    print("  GET  /search   - 组件搜索")
    print("  GET  /kinds    - 组件类别列表")
    print("  POST /reset    - 重置会话")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)
