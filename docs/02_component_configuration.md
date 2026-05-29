# 02 — Component & Project Configuration（组件与项目级配置）

此文档说明 mlblack 新增的 ProjectConfig/Registry/Builder 链路的设计、用途与示例（P1 优先级）。目标是让使用者能像 nsgablack 一样通过项目级 `config.py` 聚合 registry，并在运行时用 key 选择组件。

## 背景与目标

- mlblack 已有 `Spec` 与 `Builder`（`assembly/spec.py`、`assembly/builders.py`），但缺少框架级的 `ProjectConfig` 聚集入口，导致每个示例重复实现 registry。
- 本补充旨在：
  - 提供 `TrainerProjectConfig`（框架层）以聚集各层 registry；
  - 统一脚手架生成 `config.py`（项目级），包含 `get_project_config()`；
  - 规范 `registry` / `build_*` / `register_*` 的使用模式，便于运行时按 key 构建组件。

## 核心概念（简述）

- ComponentSpec（或 `_ComponentSpec`）：在 `assembly/project_config.py` 中用于描述 registry 条目（key + params）。
- XxxRegistry：每个层（problem/representation/adapter/bias/capability/pipeline/preset）有一个 Registry，包含若干 ComponentSpec。
- TrainerProjectConfig：Dataclass，聚集所有 XxxRegistry 实例。
- build_* / register_*：框架级的构建与注册入口（按 key 匹配并调用已注册的 builder）。

## 使用说明（代码示例）

1) 在项目中生成脚手架（示例，PowerShell）：

```powershell
# 在 mlblack 源代码根目录下（已集成 scaffold_legacy）
python -c "from mlblack.project.scaffold_legacy import init_project; init_project('my_project')"
# 或者在交互环境中调用 init_project(tmpdir)
```

脚手架会在 `my_project/` 生成 `config.py` 以及每个层的 `*/config.py`（例如 `problem/config.py`, `representation/config.py` 等）。

2) 项目级 `config.py`（脚手架示例）会包含：

```python
from mlblack.assembly import TrainerProjectConfig

def get_project_config() -> TrainerProjectConfig:
    from problem.config import get_problem_registry
    from representation.config import get_representation_registry
    from adapter.config import get_adapter_registry
    from bias.config import get_bias_registry
    from capabilities.config import get_capability_registry
    from pipeline.config import get_pipeline_registry
    from assembly.preset_registry import get_preset_registry

    return TrainerProjectConfig(
        problems=get_problem_registry(),
        representations=get_representation_registry(),
        adapters=get_adapter_registry(),
        biases=get_bias_registry(),
        capabilities=get_capability_registry(),
        pipelines=get_pipeline_registry(),
        presets=get_preset_registry(),
    )
```

3) 运行时按 key 构建组件（与 nsgablack 对齐）：

```python
from mlblack.project.config import get_project_config
from mlblack.assembly import build_problem, build_adapter, build_representation

cfg = get_project_config()
problem = build_problem(cfg.problems, 'example')
representation = build_representation(cfg.representations, 'mlp')
adapter = build_adapter(cfg.adapters, 'gradient_descent')
```

4) 注册自定义 builder（框架扩展）

在你自己的模块中注册 builder：

```python
from mlblack.assembly import register_problem_builder

def my_problem_builder(params: dict):
    return MyProblem(**params)

register_problem_builder('my_problem', my_problem_builder)
```

这会使 `build_problem(..., 'my_problem')` 生效。

## 迁移指南

- 若当前项目已有各层 `config.py`（手写），推荐：
  1. 将现有 registry 的返回值迁移为 `ProblemSpec` / `AdapterSpec` 等（key + params）。
  2. 在项目根 `config.py` 中实现 `get_project_config()` 返回 `TrainerProjectConfig` 聚合所有 `get_*_registry()` 的结果。
  3. 可选：把常用 preset 放入 `assembly/preset_registry.py`。

- 注意：大对象不要写入 runtime context；请继续使用 Snapshot/Artifact 存储大对象（与 framework 协议一致）。

## 常见问题（FAQ）

- Q: 项目级配置是否必须使用？
  - A: 推荐使用。它允许统一管理组件并简化运行时选择；但短小 demo 仍可保留本地 registry。

- Q: 如何在 CI 中验证 registry 的完整性？
  - A: 增加简单的 import 检查脚本：`from project.config import get_project_config; cfg = get_project_config()` 并断言所需 key 存在。

## 下一步建议

- 在 `mlblack/docs` 补充使用示例和迁移脚本（已开始，P1）。
- 可选：实现 `assembly.builders.build_trainer(problem_key=..., adapter_key=...)` 的高级入口（P2），方便命令行/CI 调用。

---

如需我把这份文档转换为 Word (`.docx`) 并放到仓库根目录或特定位置，我可以继续：会生成 `mlblack/docs/02_component_configuration.docx`（需要安装 `python-docx` 或者我可以生成 markdown + 简单转换脚本）。
