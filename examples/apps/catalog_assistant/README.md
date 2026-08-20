# Catalog Assistant 应用

这是消费 `mlblack` 与 `nsgablack` Catalog 的 Flask 应用，不是 Trainer/Solver Case。

```powershell
cd mlblack
$env:AI_API_KEY = "sk-你的key"
python -m examples.apps.catalog_assistant.server
```

如果应用需要触发训练或优化，应调用正式 Project/Case surface，并消费版本化结果信封。
