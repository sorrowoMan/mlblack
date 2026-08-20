# 应用示例

这里保存使用框架能力的应用服务器与交互程序。它们不是 Solver/Trainer Case，
因此不参与 Project/Case 资源授权、生命周期与结果信封。

- `catalog_assistant/`：组件目录检索助手。
- `mall_ai_assistant/`：商品检索与对话应用。

当应用需要训练或优化时，应通过正式 Project/Case API 调用独立 Case，而不是把
Web 服务器伪装成 Case。
