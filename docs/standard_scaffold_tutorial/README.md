# 标准脚手架教程（mlblack）

本教程与 `nsgablack` 保持同一套统一框架口径：

- 共享 Project / Case / Scaffold / L0 substrate
- 编排归属 substrate
- Case 主入口按 `.case kind` 解析
- 每个 Case 仅一个 `pipeline/main.py` 主入口，内部由 slot operator 组合

## 推荐阅读顺序

1. `00_assembly_api_reference.md`
2. `01_create_and_run.md`
3. `02_component_configuration.md`
4. `03_model_composition_and_io_contract.md`
5. `04_nsgablack_orchestration_and_resource_layers.md`
6. `05_symbolic_nested_case.md`
7. `06_validation_catalog_artifacts.md`
8. `07_benchmark_dashboard_resource.md`
9. `08_complex_pattern_catalog.md`
10. `09_slot_kernel_minimal_spec.md`
11. `10_custom_adapter.md`
12. `11_custom_bias.md`
13. `12_custom_plugin_hooks.md`
14. `13_pipeline_orchestration_and_component_design.md`

## 统一工作流

脚手架 CLI 与 nsgablack 共用：

```powershell
python -m nsgablack project new my_project
python -m nsgablack project add-case my_case --type trainer --framework mlblack
python -m nsgablack project add-component --case my_case --kind pipeline --slot codec --name my_codec
```

Same substrate, different semantics:

- `nsgablack`：优化/搜索语义
- `mlblack`：ML 数据/模型/训练/产物语义
