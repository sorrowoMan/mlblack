# Mall AI Assistant

商城 AI 悬浮导购 —— 右下角聊天按钮，前后端打通。

## 结构

```
Python 后端 (`mlblack/examples/apps/mall_ai_assistant/`)
├── config.py             # API Key + 角色 Prompt
├── cloudemall.db         # SQLite 本地商品库 (47件 + 32分类)
├── build_sqlite.py       # 重建 SQLite 库
├── product_data.py       # 商品搜索 (MySQL → SQLite 回退)
├── ai_engine.py          # 搜商品 → 拼上下文 → 调 LLM
├── server.py             # Flask 服务 :5000
└── test_client.py        # 命令行测试

Vue 前端 (cloudmall-microservice/mis-ui-web/)
├── src/components/FloatingAiChat.vue  # 悬浮聊天组件
├── src/layout/mall/index.vue          # 已嵌入组件
└── vite.config.ts                     # 已添加 /ai 代理
```

## 启动方式

```powershell
# 1. 启动 AI 后端 (Python)
cd mlblack
$env:AI_API_KEY = "sk-你的key"
python -m examples.apps.mall_ai_assistant.server

# 2. 启动前端 (Vue)
cd cloudmall-microservice\mis-ui-web
pnpm dev
```

然后打开 `http://localhost:5174/mall/home`，右下角出现 AI 导购按钮。

## 数据流

```
用户点击悬浮球 → 输入问题 → POST /ai/chat
  → Vite proxy → Python Flask :5000/chat
    → product_data.py 搜商品 (MySQL or SQLite)
    → ai_engine.py 拼 Prompt → DeepSeek API
    → 返回 {reply, products}
  → Vue 渲染聊天气泡 + 商品卡片
```

## 生产部署

后端网关添加路由规则：
```
/ai/*  →  http://python-ai-service:5000/*
```
