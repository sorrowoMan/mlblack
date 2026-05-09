from __future__ import annotations

from typing import Any, Mapping

_KIND_LABEL_ZH: dict[str, str] = {
    "family": "家族",
    "preset": "预设",
    "trainer": "训练器",
    "head": "输出头",
    "component": "组件",
    "provider": "供能器",
    "plugin": "插件",
    "bias": "偏置",
    "pipeline": "流水线",
    "numericizer": "数值化器",
    "doc": "文档",
    "example": "示例",
}

_FAMILY_I18N: dict[str, dict[str, Any]] = {
    "linear": {
        "title_zh": "线性家族",
        "summary_zh": "固定线性函数骨架的拟合家族，结构先验给定，训练重点是参数求解。",
        "use_when_zh": (
            "需要稳定、可解释、轻量的监督学习基线时。",
            "需要 warm start、incremental 或闭式求解友好的训练骨架时。",
        ),
    },
    "neural": {
        "title_zh": "神经网络家族",
        "summary_zh": "固定网络骨架加梯度优化的拟合家族，适合继续挂采样、加权和状态信号等神经机制。",
        "use_when_zh": (
            "需要固定网络骨架并主要通过梯度优化学习参数时。",
            "需要接入 batch policy、gradient norm、dropout uncertainty 一类神经专属机制时。",
        ),
    },
    "tree_ensemble": {
        "title_zh": "树集成家族",
        "summary_zh": "以 bagging、forest 或同类聚合为主的树集成家族，重点是多树汇聚而不是链式残差修正。",
        "use_when_zh": (
            "需要稳健的树集成基线但不需要 boosting 式逐轮残差更新时。",
            "需要对子采样、样本加权和集成汇总机制进行复用时。",
        ),
    },
    "tree_boosting": {
        "title_zh": "树提升家族",
        "summary_zh": "以逐轮加法残差修正为核心的树提升家族，完整训练骨架由 boosting 流程定义。",
        "use_when_zh": (
            "需要强 tabular 基线并希望通过 boosting 逐步修正误差时。",
            "需要保留 boosting family 的主训练语义而不是把它误当成普通组件时。",
        ),
    },
    "symbolic": {
        "title_zh": "符号学习家族",
        "summary_zh": "把结构搜索和参数拟合共同纳入训练目标的家族，候选池、grammar 与 structure engine 都是主骨架的一部分。",
        "use_when_zh": (
            "需要显式搜索表达式结构，而不是只在固定形式上调参数时。",
            "需要 candidate pool、primitive registry 或结构搜索预算成为一等公民时。",
        ),
    },
}

_HEAD_I18N: dict[str, dict[str, Any]] = {
    "point": {
        "title_zh": "单点输出头",
        "summary_zh": "输出单个中心估计值的 head，最适合标准回归或点预测任务。",
        "use_when_zh": ("需要单个预测值而不是区间或分布输出时。",),
    },
    "interval": {
        "title_zh": "区间输出头",
        "summary_zh": "输出上下界的区间 head，用来表达覆盖区间或不确定性边界。",
        "use_when_zh": ("需要 lower/upper 边界、coverage 或 uncertainty band 时。",),
    },
}

_PRESET_I18N: dict[str, dict[str, Any]] = {
    "ridge": {
        "title_zh": "Ridge 线性回归",
        "summary_zh": "线性家族的 ridge 预设：固定仿射骨架，默认输出单点预测，参数以后端闭式或近闭式方式拟合。",
        "use_when_zh": (
            "需要可解释、轻量、稳定的线性回归基线时。",
            "需要 warm start / incremental 友好的 tabular 监督学习起点时。",
        ),
    },
    "mlp_torch": {
        "title_zh": "Torch 多层感知机",
        "summary_zh": "神经网络家族的 Torch MLP 预设：固定网络骨架加梯度优化，默认输出单点回归，并可挂神经专属机制栈。",
        "use_when_zh": (
            "需要可扩展的神经网络训练骨架并希望接入 batch sampling、state signal 或自定义训练策略时。",
            "需要 Torch 后端、设备控制或更细的训练循环可塑性时。",
        ),
    },
    "sklearn_mlp": {
        "title_zh": "sklearn 多层感知机",
        "summary_zh": "神经网络家族的 sklearn MLP 预设：保留 neural family 语义，但使用更轻量的 sklearn 后端。",
        "use_when_zh": (
            "需要快速验证 neural family 脚手架能否装起来而不立刻引入完整 Torch 训练循环时。",
            "需要 CPU 上较轻量的多层感知机基线时。",
        ),
    },
    "xgboost": {
        "title_zh": "XGBoost 梯度提升树",
        "summary_zh": "树提升家族的 XGBoost 预设：以 boosting 主骨架组织训练，默认输出单点预测。",
        "use_when_zh": (
            "需要强 tabular 基线并希望保留 tree boosting family 的完整训练语义时。",
            "需要把样本加权、子采样和提升式残差更新一起纳入正式骨架时。",
        ),
    },
    "random_forest": {
        "title_zh": "随机森林",
        "summary_zh": "树集成家族的随机森林预设：通过多树 bagging 与特征随机化形成稳健的点预测基线。",
        "use_when_zh": (
            "需要稳健的树集成基线并希望较少调参时。",
            "需要利用树集成的特征子采样而不是 boosting 链式更新时。",
        ),
    },
    "extra_trees": {
        "title_zh": "极端随机树",
        "summary_zh": "树集成家族的极端随机树预设：比随机森林更强调随机切分，适合作为快速集成基线。",
        "use_when_zh": (
            "需要比随机森林更随机、更快的树集成探索时。",
            "需要在 tree ensemble family 内比较不同随机化强度时。",
        ),
    },
    "bagging": {
        "title_zh": "Bagging 集成",
        "summary_zh": "树集成家族的 Bagging 预设：把基学习器复制成多个并做聚合，强调方差降低与稳健性。",
        "use_when_zh": (
            "需要通用 bagging 集成骨架时。",
            "需要把聚合视角正式放进 family / preset 体系时。",
        ),
    },
    "adaboost": {
        "title_zh": "AdaBoost 集成",
        "summary_zh": "树集成家族的 AdaBoost 预设：通过逐轮重加权强化难样本，保留点预测主语义。",
        "use_when_zh": (
            "需要比普通 bagging 更强调难例重加权时。",
            "需要把 boosting-like 重加权作为 tree ensemble preset 来管理时。",
        ),
    },
    "symbolic": {
        "title_zh": "统一符号学习",
        "summary_zh": "符号学习家族的统一预设：结构搜索与参数拟合共同参与训练，支持 point 与 interval 语义扩展。",
        "use_when_zh": (
            "需要正式的 symbolic family，而不是把结构搜索埋回普通 trainer 里时。",
        ),
    },
    "symbolic_stagewise": {
        "title_zh": "阶段式符号学习（兼容）",
        "summary_zh": "旧阶段式 symbolic 训练入口，仍保留为兼容预设，但语义上应逐步回收到统一 symbolic family。",
        "use_when_zh": ("需要兼容旧阶段式 symbolic 配置或回归测试时。",),
    },
    "symbolic_torch": {
        "title_zh": "Torch 符号学习（兼容）",
        "summary_zh": "旧 Torch symbolic 训练入口，保留为兼容预设，用于迁移到统一 symbolic family 之前的过渡。",
        "use_when_zh": ("需要兼容旧 Torch symbolic 路径时。",),
    },
    "symbolic_torch_interval": {
        "title_zh": "Torch 区间符号学习（兼容）",
        "summary_zh": "旧 Torch interval symbolic 入口，保留为兼容预设，主要服务于迁移与等价回归。",
        "use_when_zh": ("需要兼容旧 interval symbolic 路径时。",),
    },
}

_COMPONENT_I18N: dict[str, dict[str, Any]] = {
    "bias.l2_scale": {
        "title_zh": "L2 缩放偏置",
        "summary_zh": "调节有效 L2 正则强度的偏置组件，本身不改变 family 骨架。",
        "use_when_zh": ("需要在不改动主训练骨架的前提下调节 L2 正则强度时。",),
    },
    "bias.noop": {
        "title_zh": "空操作偏置",
        "summary_zh": "不施加额外偏置的占位组件，用来保持装配面一致。",
        "use_when_zh": ("需要显式保留 bias 插槽但当前不想施加任何偏置时。",),
    },
    "aggregation.ensemble_summary": {
        "title_zh": "集成汇总视图",
        "summary_zh": "记录聚合输出、活跃信号和集成结构摘要的运行时组件，不单独定义 family。",
        "use_when_zh": ("需要把 ensemble / aggregation 结构以组件形式暴露给报告、诊断或后续流程时。",),
    },
    "sample_weighting.loss_adaptive": {
        "title_zh": "损失自适应样本加权",
        "summary_zh": "根据损失、不确定性或难度信号动态调整样本权重的组件。",
        "use_when_zh": ("需要 hard example mining、curriculum 或 focal-style weighting 时。",),
    },
    "sampling.batch_priority_subsample": {
        "title_zh": "优先级批采样",
        "summary_zh": "按运行时得分选择 batch 的采样组件，强调真正的 DataLoader / Sampler 级策略而不是预训练静态裁切。",
        "use_when_zh": ("需要基于 loss、gradient norm 或 uncertainty 做优先级 batch 采样时。",),
    },
    "sampling.row_feature_subsample": {
        "title_zh": "行/特征子采样",
        "summary_zh": "对样本行或特征列做局部裁剪的组件，用来提升效率而不改变 preset 身份。",
        "use_when_zh": ("需要在保留原 family 的前提下做样本或特征子采样时。",),
    },
    "state_signal_view.gradient_norm": {
        "title_zh": "梯度范数状态信号",
        "summary_zh": "面向神经网络训练的梯度范数观测组件，可为采样、加权和诊断提供运行态信号。",
        "use_when_zh": ("需要把 per-sample gradient magnitude 暴露给上层策略时。",),
    },
    "state_signal_view.prediction_residual": {
        "title_zh": "预测残差状态信号",
        "summary_zh": "暴露预测值、残差和相关损失视图的状态信号组件，供后续策略消费。",
        "use_when_zh": ("需要基于 prediction / residual / loss 组织后续机制时。",),
    },
}

_PROVIDER_I18N: dict[str, dict[str, Any]] = {
    "batch_evaluation_proxy_provider": {
        "title_zh": "批量评估代理",
        "summary_zh": "把 mlblack 的批量评估能力暴露给外部控制平面的 provider。",
        "use_when_zh": ("需要从外部 workflow 或 solver 侧复用 mlblack 批量评估路径时。",),
    },
    "decision_evaluation_bridge": {
        "title_zh": "决策评估桥",
        "summary_zh": "把外部决策对象解码并桥接到 mlblack 评估路径的 provider。",
        "use_when_zh": ("需要让外部 solver 的 decision 结构进入 mlblack evaluator 时。",),
    },
}

_PLUGIN_I18N: dict[str, dict[str, Any]] = {
    "experiment_tracker": {
        "title_zh": "实验追踪器",
        "summary_zh": "记录运行、事件与指标的插件，用于提供可追溯实验观测面。",
        "use_when_zh": ("需要记录 run / event / metric，并保留可追溯实验轨迹时。",),
    },
    "metric_guard": {
        "title_zh": "指标守卫",
        "summary_zh": "监控关键指标并在越界或异常时触发保护逻辑的插件。",
        "use_when_zh": ("需要在训练或评估指标异常时尽早阻断、告警或降级时。",),
    },
    "noop": {
        "title_zh": "空操作插件",
        "summary_zh": "不产生副作用的占位插件，用来保持流程编排面一致。",
        "use_when_zh": ("需要保留插件插槽但当前不想执行任何副作用时。",),
    },
    "report_writer": {
        "title_zh": "报告写出器",
        "summary_zh": "在流程尾部输出报告与摘要工件的插件。",
        "use_when_zh": ("需要在训练完成后稳定地产出报告和摘要工件时。",),
    },
    "reproducibility": {
        "title_zh": "可复现性插件",
        "summary_zh": "设置随机种子并记录复现实验元数据的插件。",
        "use_when_zh": ("需要固定随机性并留下复现实验所需信息时。",),
    },
    "runtime_resource_cleanup": {
        "title_zh": "运行时资源清理",
        "summary_zh": "在结束或异常时清理缓存、句柄和运行时资源的插件。",
        "use_when_zh": ("需要确保流程退出时释放图缓存、文件句柄和其他运行时资源时。",),
    },
    "trainer_state_checkpoint": {
        "title_zh": "训练器状态检查点",
        "summary_zh": "负责持久化 trainer_state 的插件，用来支持恢复、审计与断点续训。",
        "use_when_zh": ("需要把 trainer_state 正式落盘并支持恢复或审计时。",),
    },
}

_BIAS_I18N: dict[str, dict[str, Any]] = {
    "l2_scale": {
        "title_zh": "L2 缩放偏置",
        "summary_zh": "缩放有效 L2 正则强度的 bias 条目。",
        "use_when_zh": ("需要在 bias 表面调节 L2 正则强度时。",),
    },
    "noop": {
        "title_zh": "空操作偏置",
        "summary_zh": "不做任何处理的 bias 条目。",
        "use_when_zh": ("需要兼容 bias 装配面但当前不施加偏置时。",),
    },
}

_PIPELINE_I18N: dict[str, dict[str, Any]] = {
    "identity": {
        "title_zh": "恒等流水线",
        "summary_zh": "不改变输入的流水线条目，用于保留最简数据路径。",
        "use_when_zh": ("需要保留 pipeline 插槽但当前不做额外变换时。",),
    },
    "zscore": {
        "title_zh": "Z-Score 标准化流水线",
        "summary_zh": "对输入做标准化处理的流水线条目。",
        "use_when_zh": ("需要在训练前做标准化预处理时。",),
    },
}

_NUMERICIZER_I18N: dict[str, dict[str, Any]] = {
    "default": {
        "title_zh": "默认数值化器",
        "summary_zh": "把强类型样本编码成数值特征的默认 numericizer。",
        "use_when_zh": ("需要通用的 sample -> numeric 编码入口时。",),
    },
}


def _strip_prefix(key: str) -> str:
    text = str(key).strip()
    if ":" in text:
        return text.split(":", 1)[1]
    return text


def _kind_label(kind: str) -> str:
    return _KIND_LABEL_ZH.get(str(kind).strip().lower(), str(kind).strip() or "条目")


def _fallback_title_zh(kind: str, key: str, name: str) -> str:
    label = _kind_label(kind)
    base = str(name).strip() or _strip_prefix(key)
    return f"{label}：{base}"


def _fallback_summary_zh(kind: str, key: str) -> str:
    label = _kind_label(kind)
    return f"已注册的{label}条目“{_strip_prefix(key)}”。"


def _trainer_i18n_info(key: str) -> dict[str, Any]:
    base = dict(_PRESET_I18N.get(key, {}))
    if not base:
        return {}
    return {
        "title_zh": f"{str(base.get('title_zh', key))}训练器",
        "summary_zh": f"兼容 trainer 表面，对应预设“{str(base.get('title_zh', key))}”。",
        "use_when_zh": ("需要兼容旧 direct trainer / trainer_key 入口时。",),
    }


def build_entry_i18n_fields(
    *,
    kind: str,
    key: str,
    name: str,
    summary: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kind_key = str(kind).strip().lower()
    bare_key = _strip_prefix(key).strip().lower()
    md = dict(metadata or {})

    if kind_key == "family":
        info = dict(_FAMILY_I18N.get(bare_key, {}))
    elif kind_key == "head":
        info = dict(_HEAD_I18N.get(bare_key, {}))
    elif kind_key == "preset":
        info = dict(_PRESET_I18N.get(bare_key, {}))
    elif kind_key == "trainer":
        info = _trainer_i18n_info(bare_key)
    elif kind_key == "component":
        info = dict(_COMPONENT_I18N.get(bare_key, {}))
    elif kind_key == "provider":
        info = dict(_PROVIDER_I18N.get(bare_key, {}))
    elif kind_key == "plugin":
        info = dict(_PLUGIN_I18N.get(bare_key, {}))
    elif kind_key == "bias":
        info = dict(_BIAS_I18N.get(bare_key, {}))
    elif kind_key == "pipeline":
        info = dict(_PIPELINE_I18N.get(bare_key, {}))
    elif kind_key == "numericizer":
        info = dict(_NUMERICIZER_I18N.get(bare_key, {}))
    elif kind_key == "doc":
        info = {
            "title_zh": f"文档：{name}",
            "summary_zh": f"文档页面：{_strip_prefix(key)}",
        }
    elif kind_key == "example":
        info = {
            "title_zh": f"示例：{name}",
            "summary_zh": f"示例脚本：{_strip_prefix(key)}",
        }
    else:
        info = {}

    title_zh = str(info.get("title_zh") or md.get("title_zh") or "").strip()
    summary_zh = str(info.get("summary_zh") or md.get("summary_zh") or "").strip()
    use_when_zh = tuple(str(v).strip() for v in tuple(info.get("use_when_zh", md.get("use_when_zh", ()))) if str(v).strip())

    payload: dict[str, Any] = {}
    payload["title_zh"] = title_zh or _fallback_title_zh(kind_key, key, name)
    payload["summary_zh"] = summary_zh or _fallback_summary_zh(kind_key, key)
    if use_when_zh:
        payload["use_when_zh"] = use_when_zh
    return payload

