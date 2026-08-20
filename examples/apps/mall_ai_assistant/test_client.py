"""测试客户端 —— 不需要启动服务器，直接命令行测试 AI 对话"""
from .ai_engine import chat


def main():
    print("=" * 50)
    print("  Mall AI Assistant - 命令行测试")
    print("  输入 /exit 退出, /reset 清空对话")
    print("=" * 50)

    history = []

    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query == "/exit":
            break
        if query == "/reset":
            history = []
            print("[对话已清空]")
            continue

        print("AI: ", end="", flush=True)
        result = chat(user_query=query, history=history)

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result["reply"]})

        print(result["reply"])
        if result["products"]:
            print(f"\n  [相关商品: {len(result['products'])} 件]")


if __name__ == "__main__":
    main()
