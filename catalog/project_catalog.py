from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # py>=3.11
    import tomllib as _toml
except Exception:  # pragma: no cover
    try:  # py<3.11
        import tomli as _toml  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        _toml = None

from .registry import CatalogEntry


def _catalog_markers(root: Path) -> tuple[Path, ...]:
    return (
        root / ".mlblack-project",
        root / "catalog" / "entries.toml",
    )


def _has_catalog_entries_dir(root: Path) -> bool:
    entries_dir = root / "catalog" / "entries"
    return entries_dir.is_dir() and any(entries_dir.glob("*.toml"))


def find_project_root(start: Path | str | None) -> Path | None:
    candidate = Path(start).resolve() if start else Path.cwd().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for current in (candidate, *candidate.parents):
        if any(path.exists() for path in _catalog_markers(current)) or _has_catalog_entries_dir(current):
            return current
    return None


def _coerce_strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Mapping):
        return tuple(str(key).strip() for key in value.keys() if str(key).strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_coerce_strings(item))
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else ()


def _coerce_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_path(raw_path: str | None, *, project_root: Path, catalog_file: Path) -> str | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (catalog_file.parent / path).resolve()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _entry_from_mapping(raw: Mapping[str, Any], *, project_root: Path, catalog_file: Path) -> CatalogEntry | None:
    kind = str(raw.get("kind", "") or "").strip().lower()
    key = str(raw.get("key", "") or "").strip()
    name = str(raw.get("name", raw.get("title", "")) or "").strip()
    if not key and kind and name:
        key = f"{kind}:{name}"
    if not key or not kind:
        return None
    if not name:
        name = key.partition(":")[2].strip() or key

    metadata = _coerce_mapping(raw.get("metadata"))
    metadata.setdefault("project_catalog_file", catalog_file.relative_to(project_root).as_posix())

    fields = _coerce_mapping(raw.get("fields"))
    relations = _coerce_mapping(raw.get("relations"))
    tags = tuple(sorted({*{str(tag).strip() for tag in _coerce_strings(raw.get("tags")) if str(tag).strip()}, "project"}))

    return CatalogEntry(
        key=key,
        kind=kind,
        name=name,
        source=str(raw.get("source", "project") or "project").strip() or "project",
        path=_normalize_path(raw.get("path"), project_root=project_root, catalog_file=catalog_file),
        tags=tags,
        summary=str(raw.get("summary", "") or "").strip(),
        metadata=metadata,
        fields=fields,
        relations=relations,
    )


def _iter_entry_blocks(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload.get("entry"), list):
        for item in payload["entry"]:
            if isinstance(item, Mapping):
                yield item
    if isinstance(payload.get("entries"), list):
        for item in payload["entries"]:
            if isinstance(item, Mapping):
                yield item
    if {"key", "kind"} <= set(payload.keys()):
        yield payload


def _load_catalog_file(path: Path, *, project_root: Path) -> tuple[CatalogEntry, ...]:
    if _toml is None or not path.exists() or not path.is_file():
        return ()
    try:
        payload = _toml.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ()
    if not isinstance(payload, Mapping):
        return ()
    entries: list[CatalogEntry] = []
    for item in _iter_entry_blocks(payload):
        entry = _entry_from_mapping(item, project_root=project_root, catalog_file=path)
        if entry is not None:
            entries.append(entry)
    return tuple(entries)


def _catalog_files(project_root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    legacy = project_root / "catalog" / "entries.toml"
    if legacy.is_file():
        files.append(legacy)
    entries_dir = project_root / "catalog" / "entries"
    if entries_dir.is_dir():
        files.extend(sorted(entries_dir.glob("*.toml")))
    return tuple(files)


def load_project_entries(project_root: Path | str) -> tuple[CatalogEntry, ...]:
    root = Path(project_root).resolve()
    entries: list[CatalogEntry] = []
    for path in _catalog_files(root):
        entries.extend(_load_catalog_file(path, project_root=root))
    return tuple(entries)
