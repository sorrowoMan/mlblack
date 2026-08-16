# 11. 自定义 Bias（mlblack 详细实战）

mlblack 的 Bias 常用于训练语义中的软引导，例如：

- objective 权重偏置
- 参数尺度偏置
- 分支策略偏置

## 1. 创建 Bias 组件

```powershell
python -m nsgablack project add-component --case my_trainer --kind bias --name my_training_bias
```

## 2. 设计原则

- 偏置是 soft guidance，不是硬约束替代
- 不应静默改变任务定义
- 不应拥有编排/资源权限

## 3. 典型场景

### 3.1 多目标训练权重偏置

例如 point + interval 的组合中，早期强调稳定性，后期强调精度。

### 3.2 状态正则偏置

例如对参数范数做软惩罚，引导更稳定更新。

### 3.3 分支路由偏置

例如某些 task_kind 更倾向某类 head 或 codec。

## 4. 审计建议

至少记录：

- bias 名称与参数
- 生效阶段
- 与无 bias 基线对比
