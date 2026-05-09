from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.state.context_keys import RUN_STAGE
from experiment.dashboard import _build_deep_link_query

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

_EDGE_CANDIDATES = (
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
)
_EDGE_AVAILABLE = any(path.exists() for path in _EDGE_CANDIDATES)
pytestmark = pytest.mark.skipif(not _EDGE_AVAILABLE, reason="Microsoft Edge is required for experiment dashboard E2E tests.")

_ROOT = Path(__file__).resolve().parents[1]


def _materialize_experiment_run(
    db_path: Path,
    *,
    run_name: str,
    trainer_name: str,
    artifact_id: str,
    surface_key: str,
) -> str:
    from core.flow_experiment_tracker import ExperimentTrackerCapability

    cap = ExperimentTrackerCapability(db_path=str(db_path), namespace="ut_experiment_e2e")
    ctx: dict[str, object] = {
        "run_name": run_name,
        "surface_key": surface_key,
        "context_refs": {RUN_STAGE: "report"},
        "snapshot_count": 2,
        "output_dir": str(db_path.parent / "out"),
        "trainer": SimpleNamespace(name=trainer_name),
        "report": {
            "run_name": run_name,
            "trainer_name": trainer_name,
            "training": {
                "requested_init": {"mode": "warm_start"},
                "task_signature": {
                    "symbolic_family_signature": f"sig_{run_name}",
                    "metadata": {
                        "symbolic_family": {
                            "search_family_signature_contracts": [
                                {"mechanism_key": "beam_selection", "consume": ["gradient_signal"]}
                            ]
                        }
                    },
                },
            },
            "artifact": {
                "artifact_id": artifact_id,
                "symbolic_artifact_schema": {
                    "head_semantics": {"task": "interval"},
                    "complexity_metrics": {"term_count": 3},
                    "stability_metrics": {
                        "fold_count": 3,
                        "fold_summary": {
                            "rmse_mean": 0.31,
                            "rmse_std": 0.02,
                            "coverage_error_mean": 0.04,
                        },
                        "rmse_mean": 0.31,
                        "rmse_std": 0.02,
                        "coverage_error_mean": 0.04,
                    },
                },
            },
        },
    }
    cap.on_flow_start(ctx)
    cap.on_flow_finish(ctx)
    tracker = dict(ctx.get("experiment_tracker", {}))
    return str(tracker.get("run_id", ""))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_ready(base_url: str, *, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url, timeout=2.0) as response:
                if int(getattr(response, "status", 200)) < 500:
                    return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            time.sleep(0.5)
    raise AssertionError(f"mlblack experiment ui did not become reachable: {base_url} ({last_error})")


@contextlib.contextmanager
def _running_experiment_ui(db_path: Path, *extra_args: str):
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile(prefix="mlblack_experiment_ui_", suffix=".log", delete=False)
    log_path = Path(log_file.name)
    log_file.close()
    command = [
        sys.executable,
        "-m",
        "mlblack",
        "experiment",
        "ui",
        "--db",
        str(db_path),
        "--port",
        str(port),
        "--headless",
        *extra_args,
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            command,
            cwd=_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            creationflags=creationflags,
        )
    try:
        _wait_for_http_ready(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=10)
        with contextlib.suppress(OSError):
            log_path.unlink()


def _wait_for_deep_link(page, *, timeout_ms: int = 60000) -> None:
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        page.wait_for_timeout(700)
        locator = page.get_by_label("直达链接")
        if locator.count() == 1:
            value = locator.input_value()
            if value.strip():
                return
    raise AssertionError("直达链接输入框未能按时出现")


def _wait_for_body_text(page, expected: str, *, timeout_ms: int = 60000) -> None:
    deadline = time.time() + (timeout_ms / 1000.0)
    last_text = ""
    while time.time() < deadline:
        page.wait_for_timeout(700)
        last_text = page.locator("body").inner_text()
        if expected in last_text:
            return
    raise AssertionError(f"expected text not found: {expected!r}\nLast body text:\n{last_text[:4000]}")


def test_experiment_dashboard_deep_link_roundtrip_e2e(tmp_path: Path):
    db_path = tmp_path / "experiments.sqlite3"
    selected_run_id = _materialize_experiment_run(
        db_path,
        run_name="exp_run_selected",
        trainer_name="symbolic_torch_interval",
        artifact_id="artifact_interval_selected",
        surface_key="flow:exp_selected",
    )
    other_run_id = _materialize_experiment_run(
        db_path,
        run_name="exp_run_other",
        trainer_name="xgboost",
        artifact_id="artifact_point_other",
        surface_key="flow:exp_other",
    )
    assert selected_run_id and other_run_id

    initial_query = _build_deep_link_query(
        base_params={
            "db": str(db_path),
            "limit": "200",
            "view": "run_catalog",
            "selected": f"run:{selected_run_id}",
        },
        field_filters={
            "run_status": "completed",
            "run_trainer_name": "symbolic_torch_interval",
            "run_fold_summary": "present",
            "run_surface_key": "flow:exp_selected",
            "run_family_ref": "family:symbolic",
        },
    )

    with _running_experiment_ui(db_path, "--limit", "200") as base_url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1400})
            page.goto(base_url + initial_query, wait_until="load", timeout=120000)
            _wait_for_deep_link(page)
            _wait_for_body_text(page, selected_run_id)

            deep_link = page.get_by_label("直达链接").input_value()
            assert "view=run_catalog" in deep_link
            assert "f_run_trainer_name=symbolic_torch_interval" in deep_link
            assert "f_run_surface_key=flow%3Aexp_selected" in deep_link or "f_run_surface_key=flow:exp_selected" in deep_link
            assert selected_run_id in page.locator("body").inner_text()

            page2 = browser.new_page(viewport={"width": 1600, "height": 1400})
            page2.goto(base_url + deep_link, wait_until="load", timeout=120000)
            _wait_for_deep_link(page2)
            _wait_for_body_text(page2, selected_run_id)
            assert page2.get_by_label("直达链接").input_value() == deep_link
            assert other_run_id not in page2.locator("body").inner_text()

            browser.close()
