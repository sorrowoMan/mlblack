# 04. Add Backend Guide

这篇说明后续如何新增一个 compute backend。目标是让新 backend 接入现有 `NeuralGraphCodec / Problem / Adapter / Artifact` 路线，而不是改出第二套训练框架。

## 什么时候需要新 backend

适合新增 backend：

```text
需要接入新的 tensor/autograd 执行系统。
需要利用该系统已有 GPU/JIT/distributed/kernel 能力。
需要复用该系统 optimizer/loss/artifact 工具。
```

不适合新增 backend：

```text
只是想新增一个模型结构。
只是想新增一个 problem/loss。
只是想新增一种搜索策略。
```

对应归属：

```text
新模型结构:
  NeuralGraphSpec / backend neural_lowering

新 loss:
  Problem 或 backend losses capability

新优化策略:
  Adapter

新执行系统:
  Backend
```

## 新 backend 最小目录

```text
mlblack/backends/<name>_neural/
  __init__.py
  backend.py
  capabilities/
    __init__.py
    tensor.py
    neural_lowering.py
    losses.py
    artifacts.py
```

如果支持训练，再加：

```text
    autograd.py
    optimizers.py
```

## backend.py 模板

```python
class MyBackend:
    name = "my_backend"

    def __init__(self):
        self.tensor = MyTensorCapability()
        self.lowering = MyNeuralLoweringCapability()
        self.losses = MyLossesCapability()
        self.artifacts = MyArtifactsCapability()
        self.capabilities = (
            self.tensor,
            self.lowering,
            self.losses,
            self.artifacts,
        )

    def contract(self) -> BackendContract:
        return BackendContract(
            name=self.name,
            capabilities=tuple(item.contract for item in self.capabilities),
            metadata={"family": "neural", "engine": self.name},
        )

    def parameter_layout(self, spec):
        return self.lowering.parameter_layout(spec)

    def initial_values(self, spec, *, random_seed=42):
        return self.lowering.initial_values(spec, random_seed=random_seed)

    def decode_neural_graph(self, values, spec, *, random_seed=42, context=None):
        return self.lowering.decode_neural_graph(values, spec, random_seed=random_seed)
```

## 注册 backend

在 `mlblack/backends/registry.py` 中注册：

```python
from mlblack.backends.my_backend_neural.backend import MyBackend

register_backend("my_backend", MyBackend)
```

如果有别名：

```python
aliases = {
    "my_backend_cpu": "my_backend",
}
```

## Capability contract 写法

不要写模糊能力：

```text
bad:
  provides=("training", "neural")
```

要写原子能力：

```text
good:
  provides=(
    "tensor.float_tensor",
    "neural.lowering.mlp",
    "parameters.layout",
    "parameters.init",
    "loss.mse",
    "autograd.functional.grad",
    "optimizer.sgd_step",
  )
```

## 先实现最小 route

推荐新增 backend 的顺序：

```text
1. tensor.float_tensor
2. neural.lowering.mlp
3. parameters.layout / parameters.init
4. loss.mse / metrics.regression
5. artifact.parameters.summary
6. autograd.functional.grad 或 autograd.backward
7. optimizer.sgd_step 或 optimizer.step
8. 再扩 Transformer/CNN/GNN route
```

不要第一步就做完整 Transformer。

## 两种梯度风格

Torch-style：

```text
stateful module
loss object
loss.backward()
optimizer.step()
```

需要 capability：

```text
autograd.backward
optimizer.build
optimizer.step
parameters.flat_export
```

适配：

```text
NeuralGraphBackpropAdapter
```

Functional-style：

```text
values -> loss(values)
grad(loss_fn)(values)
values - lr * grad
```

需要 capability：

```text
autograd.functional.grad
optimizer.sgd_step
```

适配：

```text
Problem.compute_functional_gradient(...)
  -> backend.autograd.functional.grad
  -> FunctionalBackpropAdapter
```

关键规则：

```text
不要把 functional backend 伪装成 torch-style backend。
不要为了兼容而声明不存在的 capability。
```

## 必须加的测试

Catalog / capability：

```python
def test_my_backend_catalog_boundaries():
    assert "my_backend" in list_backends()
    assert explain_backend_requirements(
        "my_backend",
        ("neural.lowering.mlp", "loss.mse"),
    )["ok"]
```

缺失能力：

```python
def test_my_backend_missing_torch_style_backward():
    result = explain_backend_requirements(
        "my_backend",
        ("autograd.backward", "optimizer.step"),
    )
    assert result["ok"] is False
```

Codec lowering：

```python
def test_my_backend_mlp_lowering():
    codec = NeuralGraphCodec(NeuralGraphSpec.mlp(...))
    context = {"backend.session": ComputeBackendSession(ComputeBackendSpec(name="my_backend"))}
    values = codec.init_values(context)
    model = codec.decode(values, context)
```

Trainer smoke：

```python
def test_my_backend_trainer_smoke():
    trainer = Trainer(
        problem=...,
        representation=NeuralGraphRepresentation(...),
        adapter=...,
        compute_backend=ComputeBackendSpec(name="my_backend"),
    )
    result = trainer.fit(max_steps=1)
```

## 文档更新清单

新增 backend 后，同步更新：

```text
docs/neural_graph_backend_architecture/03_backend_matrix.md
docs/neural_graph_backend_architecture/README.md
docs/transformer_learning/04_neural_graph_decoder_design.md
```

## 验收命令

```powershell
python -m compileall -q mlblack tests
python -m pytest -q tests
python -c "from mlblack.project import run_project_doctor, format_doctor_report; print(format_doctor_report(run_project_doctor('.', strict=True)))"
```

如果影响 torch neural graph，继续跑：

```powershell
python examples\cases\tiny_transformer_smoke\run_project.py --check --build-check
python examples\cases\tiny_transformer_smoke\run_project.py -- --steps 1
python examples\cases\benchmarks\run_project.py -- --steps 1 --repeats 1
```
