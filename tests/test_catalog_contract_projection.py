"""Catalog metadata publication checks; never construct or run ML components."""
from blackbase.catalog import CatalogEntry
from blackbase.context import ContextContract
from blackbase.contracts import ComponentContract

from mlblack.catalog import registry


def entry(**kwargs):
    return CatalogEntry(key="test.component", title="Component", kind="core",
                        import_path="test_component:Component", **kwargs)


def test_real_learning_contract_is_published_in_shared_fields():
    item = registry.get_catalog(refresh=True).get("core.learning")
    assert item.context_requires == ("candidate.model", "data")
    assert item.context_provides == ("feedback.objectives", "feedback.metrics")
    assert item.contract["context_requires"] == list(item.context_requires)
    assert item.context_notes


def test_explicit_object_fields_include_artifacts_and_phases_without_construction(monkeypatch):
    class Component:
        context_requires = ("data",)
        context_provides = ("candidate.model",)
        context_mutates = ("trainer.context",)
        context_cache = ("model_ref",)
        context_notes = "Reads data and publishes a model."
        artifact_requires = ("input_model",)
        artifact_provides = ("trained_model",)
        phase_in = ("fit",)
        phase_out = ("predict",)

        def __init__(self):
            raise AssertionError("Catalog inspection must not construct the component")

    monkeypatch.setattr(registry, "_resolve_import_path", lambda _: Component)
    item = registry.enrich_catalog_entry(entry())
    assert item.context_requires == ("data",)
    assert item.context_provides == ("candidate.model",)
    assert item.context_mutates == ("trainer.context",)
    assert item.context_cache == ("model_ref",)
    assert item.context_notes == ("Reads data and publishes a model.",)
    assert item.artifact_requires == ("input_model",)
    assert item.artifact_provides == ("trained_model",)
    assert item.phase_in == ("fit",)
    assert item.phase_out == ("predict",)


def test_nested_typed_context_contract_preserves_all_explicit_fields(monkeypatch):
    class Component:
        contract = ContextContract(
            requires=("data",), provides=("candidate.model",),
            mutates=("trainer.context",), cache=("model_ref",),
            artifact_requires=("input_model",), artifact_provides=("trained_model",),
            phase_in=("fit",), phase_out=("predict",), notes="Declared context.",
        )

    monkeypatch.setattr(registry, "_resolve_import_path", lambda _: Component)
    item = registry.enrich_catalog_entry(entry())
    assert item.context_requires == ("data",)
    assert item.context_provides == ("candidate.model",)
    assert item.context_mutates == ("trainer.context",)
    assert item.context_cache == ("model_ref",)
    assert item.context_notes == ("Declared context.",)
    assert item.artifact_requires == ("input_model",)
    assert item.artifact_provides == ("trained_model",)
    assert item.phase_in == ("fit",)
    assert item.phase_out == ("predict",)


def test_artifact_only_declarations_do_not_need_a_context_payload(monkeypatch):
    class Component:
        artifact_provides = ("report",)
        phase_out = ("publish",)

    monkeypatch.setattr(registry, "_resolve_import_path", lambda _: Component)
    item = registry.enrich_catalog_entry(entry())
    assert item.contract == {}
    assert item.context_provides == ()
    assert item.artifact_provides == ("report",)
    assert item.phase_out == ("publish",)


def test_component_contract_is_context_but_backend_provides_is_not(monkeypatch):
    class Component:
        contract = ComponentContract(requires=("data",), provides=("candidate.model",))

    monkeypatch.setattr(registry, "_resolve_import_path", lambda _: Component)
    typed = registry.enrich_catalog_entry(entry())
    assert typed.context_requires == ("data",)
    assert typed.context_provides == ("candidate.model",)

    monkeypatch.setattr(registry, "_resolve_import_path", lambda _: object())
    raw = {"provides": ["linear_train"], "methods": {"fit": {"inputs": ["data"]}}}
    backend = registry.enrich_catalog_entry(entry(contract=raw))
    assert backend.context_requires == backend.context_provides == ()
    assert backend.contract == raw


def test_static_declarations_merge_and_ignore_descriptor_mappings(monkeypatch):
    monkeypatch.setattr(registry, "_resolve_import_path", lambda _: object())
    raw = {
        "context_requires": [" data ", "data", {"not_a_dependency": "descriptor"}, 7],
        "context_provides": {"not_a_dependency": "descriptor"},
        "context_contract": {"requires": ("model_ref",), "provides": "candidate.model",
                             "artifact_provides": ["trained_model"], "phase_out": ["predict"]},
        "metadata": {"input": "not_a_dependency"},
    }
    item = registry.enrich_catalog_entry(entry(context_requires=("declared_input",), contract=raw))
    assert item.context_requires == ("declared_input", "data", "model_ref")
    assert item.context_provides == ("candidate.model",)
    assert item.artifact_provides == ("trained_model",)
    assert item.phase_out == ("predict",)
    assert item.contract == raw


def test_static_contract_projection_survives_unavailable_optional_import(monkeypatch):
    def unavailable(_):
        raise ImportError("optional backend is unavailable")

    monkeypatch.setattr(registry, "_resolve_import_path", unavailable)
    raw = {"context_contract": {"requires": ["model_ref"], "artifact_provides": ["report"]}}
    item = registry.enrich_catalog_entry(entry(contract=raw))
    assert item.context_requires == ("model_ref",)
    assert item.artifact_provides == ("report",)
    assert item.contract == raw
    assert "ImportError" in item.metadata["contract_error"]
