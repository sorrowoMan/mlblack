"""mlblack-owned semantic component templates for the shared scaffold writer."""

from __future__ import annotations


def render_component_template(component_name: str, component_kind: str, slot: str | None = None) -> str:
    del slot
    class_name = "".join(part.capitalize() for part in component_name.split("_") if part) or "Component"
    renderers = {
        "problem": _problem_template,
        "adapter": _adapter_template,
        "bias": _bias_template,
        "plugin": _plugin_template,
    }
    try:
        renderer = renderers[str(component_kind)]
    except KeyError as exc:
        raise ValueError(f"unsupported mlblack component kind: {component_kind}") from exc
    return renderer(class_name, component_name, component_kind)


def _problem_template(class_name: str, component_name: str, kind: str) -> str:
    return (
        f'"""Auto-generated {kind} component: {component_name}."""\n\n'
        "from mlblack.core.problem import LearningProblem\n\n\n"
        f"class {class_name}(LearningProblem):\n"
        '    """TODO: implement evaluation logic."""\n\n'
        "    def evaluate(self, candidate, context=None):\n"
        '        raise NotImplementedError("TODO: implement evaluate")\n'
    )


def _adapter_template(class_name: str, component_name: str, kind: str) -> str:
    return (
        f'"""Auto-generated {kind} component: {component_name}."""\n\n'
        "from nsgablack.adapters import AlgorithmAdapter\n\n\n"
        f"class {class_name}(AlgorithmAdapter):\n"
        '    """TODO: implement propose/update logic."""\n\n'
        "    def propose(self, control, context):\n"
        '        raise NotImplementedError("TODO: implement propose")\n\n'
        "    def update(self, control, candidates, feedback, context):\n"
        '        raise NotImplementedError("TODO: implement update")\n'
    )


def _bias_template(class_name: str, component_name: str, kind: str) -> str:
    return (
        f'"""Auto-generated {kind} component: {component_name}."""\n\n'
        "from mlblack.bias.base import OptimizationBias\n\n\n"
        f"class {class_name}(OptimizationBias):\n"
        '    """TODO: implement preference logic."""\n\n'
        "    def project_context(self, control, context):\n"
        "        del control\n"
        "        return dict(context)\n"
    )


def _plugin_template(class_name: str, component_name: str, kind: str) -> str:
    return (
        f'"""Auto-generated {kind} component: {component_name}."""\n\n'
        "from blackbase.plugin import PluginBase\n\n\n"
        f"class {class_name}(PluginBase):\n"
        '    """TODO: implement capability lifecycle hooks."""\n\n'
        "    pass\n"
    )


__all__ = ["render_component_template"]
