# -*- coding: utf-8 -*-
"""Project scaffolding -- mlblack inner-trainer project layout."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Dict

_FOLDERS = [
    "catalog",
    "problem",
    "pipeline",
    "representation",
    "adapter",
    "bias",
    "capabilities",
    "assembly",
    "assets",
    "docs",
]

_NON_PACKAGE_FOLDERS = {"catalog", "assets", "docs"}

_FOLDER_DESCRIPTIONS: Dict[str, str] = {
    "catalog": "Project-local catalog index: register discoverable local components.",
    "problem": "Learning problem layer: objective, evaluation, loss, metrics.",
    "pipeline": "Data pipeline layer: data views, numericizer, feature engineering.",
    "representation": "Model representation layer: codec, head, model-space decoding.",
    "adapter": "Optimizer strategy layer: propose/update (GD, backprop, EM).",
    "bias": "Optimization bias layer: soft preferences (L2, sparsity, branch).",
    "capabilities": "Capability layer: checkpoint, tracking, audit, report.",
    "assembly": "Trainer assembly: build_trainer, preset wiring, spec/schema.",
    "assets": "Output artifacts: charts, reports, exported files.",
    "docs": "Project documentation and design notes.",
}


def _write_file(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text(content, encoding="utf-8")


def _readme_for_folder(name: str) -> str:
    desc = _FOLDER_DESCRIPTIONS.get(name, "Module directory.")
    return dedent(f"""\
        # {name}

        - Responsibility: {desc}
        - Boundary: keep only this layer's concern; avoid cross-layer logic here.
        - Context contract (if any):
          - context_requires / context_provides / context_mutates / context_cache
        - Minimal example: keep one runnable file, or document the entry path.
        """)


def _root_readme(project_name: str) -> str:
    return dedent(f"""\
        # {project_name}

        mlblack scaffold (inner-trainer project layout).

        ## Quickstart
        1. python -m mlblack project doctor --path .
        2. python run_trainer.py
        3. python build_trainer.py

        ## Structure
        - build_trainer.py: main assembly entry
        - problem/, pipeline/, representation/
        - adapter/, bias/, capabilities/
        - assembly/, catalog/entries.toml

        ## Notes
        - Multi-stage/group/event orchestration belongs to nsgablack.
        - This scaffold provides a single inner trainer.
        - Use project doctor to validate contracts early.
        """)


def _start_here() -> str:
    return dedent("""\
        # START_HERE

        ## 1) Health Baseline
        python -m mlblack project doctor --path . --strict

        ## 2) Define the Core Layers
        - problem/: evaluate model -> produce Feedback
        - pipeline/: prepare data for the problem
        - representation/: encode/decode unknown state + head output
        - adapter/: propose/update optimization strategy

        ## 3) Wire the Assembly
        - build_trainer.py is the only assembly entry

        ## 4) Run
        python run_trainer.py

        ## 5) Verify
        python -m compileall -q .
        """)


def _problem_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        \"\"\"Example LearningProblem: supervised regression with MSE.\"\"\"

        from __future__ import annotations

        import numpy as np

        from mlblack.core.problem import LearningProblem
        from mlblack.core.types import Feedback


        class ExampleRegressionProblem(LearningProblem):
            \"\"\"Minimize MSE between model prediction and target.\"\"\"

            context_requires = ()
            context_provides = ("feedback.objectives",)
            context_mutates = ()
            context_cache = ()
            context_notes = "Computes MSE loss from model prediction vs observed target."

            def __init__(self, data, *, name="example_regression"):
                self._X = np.asarray(data.get("X", np.zeros((1, 1))), dtype=float)
                self._y = np.asarray(data.get("y", np.zeros(1)), dtype=float).ravel()
                super().__init__(name=name)

            def evaluate(self, unknown_state):
                pred = np.asarray(unknown_state, dtype=float).ravel()
                if len(pred) != len(self._y):
                    pred = np.full_like(self._y, pred[0] if len(pred) else 0.0)
                residuals = pred - self._y
                mse = float(np.mean(residuals ** 2))
                return Feedback(
                    objectives=np.array([mse]),
                    gradients=residuals,
                    constraints=np.zeros(0, dtype=float),
                )

            def describe(self):
                return {
                    "name": self.name,
                    "n_samples": len(self._y),
                    "objective": "minimize MSE",
                }
        """)


def _problem_class_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        # Problem template: copy and rename for new LearningProblems.

        from __future__ import annotations

        import numpy as np

        from mlblack.core.problem import LearningProblem
        from mlblack.core.types import Feedback


        class ProblemTemplate(LearningProblem):
            context_requires = ()
            context_provides = ("feedback.objectives",)
            context_mutates = ()
            context_cache = ()
            context_notes = "Template LearningProblem."

            def __init__(self, *, name="problem_template"):
                super().__init__(name=name)

            def evaluate(self, unknown_state):
                arr = np.asarray(unknown_state, dtype=float).ravel()
                f = float(np.sum(arr ** 2))
                return Feedback(
                    objectives=np.array([f]),
                    gradients=2.0 * arr,
                    constraints=np.zeros(0, dtype=float),
                )

            def describe(self):
                return {"name": self.name}
        """)


def _pipeline_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        \"\"\"Example data pipeline for supervised regression.\"\"\"

        from __future__ import annotations

        import numpy as np

        from mlblack.pipeline.data_views import NumericDataView


        def build_data_view(
            X,
            y,
            *,
            feature_names=(),
            target_name="target",
        ):
            return NumericDataView(
                features=X,
                target=y,
                feature_names=list(feature_names or tuple(f"x{i}" for i in range(X.shape[1]))),
                target_name=target_name,
            )
        """)


def _representation_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        \"\"\"Example ModelRepresentation: linear model with point head.\"\"\"

        from __future__ import annotations

        import numpy as np

        from mlblack.core.representation import ModelRepresentation


        class ExampleLinearRepresentation(ModelRepresentation):
            \"\"\"Simple linear model: coefficients vector with point head.\"\"\"

            context_requires = ()
            context_provides = ("model.coefficients",)
            context_mutates = ()
            context_cache = ()
            context_notes = "Encodes/decodes a linear coefficient vector."

            def __init__(self, n_features=1, *, name="linear"):
                self.n_features = max(1, int(n_features))
                super().__init__(name=name)

            def init(self, rng=None):
                rng = rng or np.random.default_rng()
                return rng.normal(0.0, 0.1, size=(self.n_features,))

            def encode(self, coefficients):
                return np.asarray(coefficients, dtype=float).ravel()

            def decode(self, encoded):
                return np.asarray(encoded, dtype=float).ravel()

            def predict(self, encoded, X):
                coef = np.asarray(encoded, dtype=float).ravel()
                return X @ coef

            def describe(self):
                return {"name": self.name, "n_features": self.n_features}
        """)


def _codec_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        \"\"\"Example Codec: float-array encode/decode (mlblack unique layer).\"\"\"

        from __future__ import annotations

        import numpy as np
        from mlblack.core.representation import ModelRepresentation


        class ExampleFloatCodec(ModelRepresentation):
            \"\"\"Identity-style codec for float arrays.\"\"\"

            context_requires = ()
            context_provides = ()
            context_mutates = ()
            context_cache = ()
            context_notes = "Minimal identity codec."

            def __init__(self, dimension=1, *, name="float_codec"):
                self.dimension = max(1, int(dimension))
                super().__init__(name=name)

            def init(self, rng=None):
                rng = rng or np.random.default_rng()
                return rng.uniform(-1.0, 1.0, size=(self.dimension,))

            def encode(self, state):
                return np.asarray(state, dtype=float).ravel()

            def decode(self, encoded):
                return np.asarray(encoded, dtype=float).ravel()
        """)


def _adapter_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        \"\"\"Example OptimizerAdapter: simple gradient descent.\"\"\"

        from __future__ import annotations

        from typing import Sequence

        import numpy as np

        from mlblack.core.adapter import OptimizerAdapter


        class ExampleGradientDescentAdapter(OptimizerAdapter):
            \"\"\"Vanilla gradient descent.\"\"\"

            context_requires = ("feedback.gradients", "candidate.unknown_state")
            context_provides = ("population.candidates",)
            context_mutates = ("adapter.current_state",)
            context_cache = ()
            context_notes = "Reads gradients, proposes next candidates via gradient step."

            def __init__(self, learning_rate=0.01, max_candidates=1, *, name="gd"):
                super().__init__(name=name)
                self.learning_rate = float(learning_rate)
                self.max_candidates = max(1, int(max_candidates))

            def propose(self, trainer, context):
                current = context.get("candidate.unknown_state")
                gradients = context.get("feedback.gradients")
                if current is None:
                    dim = getattr(getattr(trainer, "representation", None), "dimension", 1)
                    rng = np.random.default_rng()
                    return [rng.uniform(-1.0, 1.0, size=(dim,))]
                x = np.asarray(current, dtype=float).ravel()
                g = np.asarray(gradients, dtype=float).ravel()
                if len(g) != len(x):
                    g = np.zeros_like(x)
                return [x - self.learning_rate * g]

            def update(self, trainer, feedback, context):
                pass

            def describe(self):
                return {"name": self.name, "learning_rate": self.learning_rate}
        """)


def _bias_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        \"\"\"Example OptimizationBias: L2 regularization.\"\"\"

        from __future__ import annotations

        import numpy as np

        from mlblack.core.bias import OptimizationBias


        class ExampleL2Bias(OptimizationBias):
            \"\"\"Soft L2 penalty on unknown state.\"\"\"

            context_requires = ("candidate.unknown_state",)
            context_provides = ()
            context_mutates = ("feedback.objectives",)
            context_cache = ()
            context_notes = "Adds L2 penalty to the primary objective."
            requires_metrics = ()
            metrics_fallback = "none"

            def __init__(self, weight=0.01, *, name="l2_bias"):
                super().__init__(name=name)
                self.weight = float(weight)

            def compute(self, unknown_state):
                if unknown_state is None:
                    return 0.0
                return self.weight * float(np.sum(np.asarray(unknown_state) ** 2))

            def adjust_feedback(self, feedback, unknown_state, context):
                if feedback is None or not hasattr(feedback, "objectives"):
                    return feedback
                l2 = self.compute(unknown_state)
                feedback.objectives = np.asarray(feedback.objectives) + l2
                return feedback
        """)


def _capability_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        \"\"\"Example Capability: checkpoint persistence.\"\"\"

        from __future__ import annotations

        import json
        from pathlib import Path

        from mlblack.core.capability import Capability


        class ExampleCheckpointCapability(Capability):
            \"\"\"Save trainer state to disk periodically.\"\"\"

            context_requires = ()
            context_provides = ("capability.checkpoint_path",)
            context_mutates = ()
            context_cache = ()
            context_notes = "Writes trainer state as JSON every N steps."

            def __init__(self, checkpoint_dir="checkpoints", interval=10, *, name="checkpoint"):
                super().__init__(name=name)
                self.checkpoint_dir = Path(checkpoint_dir)
                self.interval = max(1, int(interval))

            def on_fit_start(self, trainer):
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            def on_step_end(self, trainer):
                step = trainer.context_store.get("step", 0)
                if step % self.interval == 0:
                    state = trainer.get_state()
                    path = self.checkpoint_dir / f"step_{step:06d}.json"
                    path.write_text(json.dumps(state, indent=2, default=str))
        """)


def _build_trainer_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        # Project build_trainer entry (inner-trainer project layout).

        from __future__ import annotations

        import argparse
        from pathlib import Path

        _HERE = Path(__file__).resolve().parent


        def build_project_trainer():
            \"\"\"Assemble and return the inner trainer.\"\"\"

            from mlblack.assembly import build_trainer as _build
            from mlblack.assembly.schema import load_scaffold_config

            config_path = _HERE / "assembly" / "scaffold.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Assembly config not found: {config_path}")

            config = load_scaffold_config(config_path)
            inner_training = dict(config.inner_training)
            trainer_spec = dict(inner_training.get("trainer", {}) or {})
            return _build(trainer_spec, None)


        def _build_parser():
            parser = argparse.ArgumentParser(
                description="Build and run the mlblack trainer scaffold.",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            )
            parser.add_argument(
                "--check",
                action="store_true",
                help="Build and validate assembly only; do not run fit().",
            )
            return parser


        def main(argv=None):
            parser = _build_parser()
            args = parser.parse_args(argv)
            trainer = build_project_trainer()
            if bool(args.check):
                problem = getattr(trainer, "problem", None)
                rep = getattr(trainer, "representation", None)
                adapter = getattr(trainer, "adapter", None)
                print(
                    "[check] assembly ok | "
                    f"problem={type(problem).__name__ if problem else 'None'} | "
                    f"representation={type(rep).__name__ if rep else 'None'} | "
                    f"adapter={type(adapter).__name__ if adapter else 'None'}"
                )
                return
            result = trainer.fit(max_steps=20)
            print(trainer.build_report())
            print(result.report)


        if __name__ == "__main__":
            main()
        """)


def _run_trainer_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        # CLI entrypoint for running the trainer scaffold.

        from __future__ import annotations

        from build_trainer import main

        if __name__ == "__main__":
            main()
        """)


def _assembly_config_template(project_name: str) -> str:
    return json.dumps(
        {
            "name": project_name,
            "features": ["x0"],
            "target": "target",
            "inner_training": {
                "pipeline": {"components": []},
                "trainer": {
                    "preset": "orthogonal_linear_point",
                    "resource_context": {"device": "cpu", "threads": 1},
                },
            },
        },
        indent=2,
    )


def _project_registry_template() -> str:
    return dedent("""\
        # -*- coding: utf-8 -*-
        # Project registry: register custom components for catalog discovery.

        from __future__ import annotations

        from mlblack.catalog.registry import get_catalog


        def register_project_components():
            catalog = get_catalog()
            # Example:
            # catalog.register("problem.my_project_regression", {
            #     "title": "My Project Regression Problem",
            #     "kind": "problem",
            #     "import_path": "problem.example_problem:ExampleRegressionProblem",
            #     "tags": ["example"],
            #     "summary": "Custom regression problem for this project.",
            # })
            pass


        if __name__ == "__main__":
            register_project_components()
        """)


def _catalog_entries_template(project_name: str) -> str:
    return dedent(f"""\
        # Catalog entries for {project_name}
        # Register discoverable components here.
        #
        # Format:
        #   [[entries]]
        #   key = "kind.component_name"
        #   title = "..."
        #   kind = "problem|pipeline|representation|adapter|bias|capability|..."
        #   import_path = "module:Class"

        [[entries]]
        key = "problem.{project_name}_regression"
        title = "{project_name.title()} Regression Problem"
        kind = "problem"
        import_path = "problem.example_problem:ExampleRegressionProblem"
        tags = ["example", "regression"]
        summary = "Custom regression problem for {project_name}."

        [[entries]]
        key = "adapter.{project_name}_gd"
        title = "{project_name.title()} Gradient Descent Adapter"
        kind = "adapter"
        import_path = "adapter.example_adapter:ExampleGradientDescentAdapter"
        tags = ["example", "gradient_descent"]
        summary = "Custom gradient descent adapter for {project_name}."
        """)


def _component_contract_template() -> str:
    return dedent("""\
        # Component Contract Card Template

        Use this card for every mlblack component.

        ## 1. Identity
        - Component key:
        - Kind:
        - Source path:

        ## 2. Responsibility
        - What this component must do:
        - What this component must NOT do:

        ## 3. Context Contract
        - context_requires:
        - context_provides:
        - context_mutates:
        - context_cache:
        - context_notes:

        ## 4. Recovery Contract
        - Implements get_state/set_state: yes/no
        - state_recovery_level: L0/L1/L2
        """)


def init_project(target_dir, *, force=False):
    """Initialize an mlblack inner-trainer project scaffold under target_dir."""
    root = Path(target_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if any(root.iterdir()) and not force:
        raise FileExistsError(f"Target directory not empty: {root}")

    project_name = root.name

    for name in _FOLDERS:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        if name not in _NON_PACKAGE_FOLDERS:
            _write_file(folder / "__init__.py", "", overwrite=force)
        _write_file(folder / "README.md", _readme_for_folder(name), overwrite=force)

    # Top-level files
    _write_file(root / "README.md", _root_readme(project_name), overwrite=force)
    _write_file(root / ".mlblack-project", "marker = mlblack-scaffold-project\n", overwrite=force)
    _write_file(root / "START_HERE.md", _start_here(), overwrite=force)
    _write_file(root / "project_registry.py", _project_registry_template(), overwrite=force)
    _write_file(root / "build_trainer.py", _build_trainer_template(), overwrite=force)
    _write_file(root / "run_trainer.py", _run_trainer_template(), overwrite=force)

    # problem/
    _write_file(root / "problem" / "example_problem.py", _problem_template(), overwrite=force)
    _write_file(root / "problem" / "template_problem.py", _problem_class_template(), overwrite=force)

    # pipeline/
    _write_file(root / "pipeline" / "example_pipeline.py", _pipeline_template(), overwrite=force)

    # representation/ (includes codec)
    _write_file(root / "representation" / "example_representation.py", _representation_template(), overwrite=force)
    _write_file(root / "representation" / "example_codec.py", _codec_template(), overwrite=force)

    # adapter/
    _write_file(root / "adapter" / "example_adapter.py", _adapter_template(), overwrite=force)

    # bias/
    _write_file(root / "bias" / "example_bias.py", _bias_template(), overwrite=force)

    # capabilities/
    _write_file(root / "capabilities" / "example_capability.py", _capability_template(), overwrite=force)

    # assembly config
    _write_file(root / "assembly" / "scaffold.json", _assembly_config_template(project_name), overwrite=force)

    # catalog
    _write_file(root / "catalog" / "entries.toml", _catalog_entries_template(project_name), overwrite=force)

    # docs
    (root / "docs" / "contracts").mkdir(parents=True, exist_ok=True)
    _write_file(root / "docs" / "contracts" / "COMPONENT_CONTRACT_TEMPLATE.md", _component_contract_template(), overwrite=force)

    return root


def create_standard_scaffold(
    path,
    *,
    name="mlblack_project",
    features=("x0",),
    target="target",
    config_overrides=None,
    exist_ok=True,
):
    """Thin wrapper -- delegates to init_project for the new scaffold."""
    root = init_project(Path(path), force=bool(exist_ok))
    return {
        "root": root,
        "config": root / "assembly" / "scaffold.json",
        "build_trainer": root / "build_trainer.py",
        "run_trainer": root / "run_trainer.py",
    }
