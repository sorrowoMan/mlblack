from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Any


def write_json_report(payload: Mapping[str, Any], *, out_dir: str, run_id: str) -> Path:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rid = str(run_id).strip() or "run_default"
    path = root / f"{rid}_summary.json"
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
