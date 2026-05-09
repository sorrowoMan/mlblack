from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

from catalog import dashboard_page as page_protocol
from catalog import dashboard_shell as shell_protocol

sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright

_EDGE_CANDIDATES = (
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
)
_EDGE_AVAILABLE = any(path.exists() for path in _EDGE_CANDIDATES)
pytestmark = pytest.mark.skipif(not _EDGE_AVAILABLE, reason="Microsoft Edge is required for catalog dashboard E2E tests.")

_ROOT = Path(__file__).resolve().parents[1]
_SECTION_IDS = (
    page_protocol.HERO_SECTION_ID,
    page_protocol.FILTER_SECTION_ID,
    page_protocol.RESULT_SECTION_ID,
    page_protocol.DETAIL_SECTION_ID,
)
_CONTROL_ROW_IDS = (
    page_protocol.PRIMARY_CONTROLS_ROW_ID,
    page_protocol.SECONDARY_CONTROLS_ROW_ID,
)
_READY_SELECTED_KEY = "preset:mlp_torch"
_NEXT_SELECTED_KEY = "preset:symbolic_torch"


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
        except Exception as exc:  # pragma: no cover - startup timing dependent
            last_error = exc
            time.sleep(0.5)
    raise AssertionError(f"catalog ui did not become reachable: {base_url} ({last_error})")


@contextlib.contextmanager
def _running_catalog_ui(*extra_args: str, env_extra: dict[str, str] | None = None):
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile(prefix="mlblack_catalog_ui_", suffix=".log", delete=False)
    log_path = Path(log_file.name)
    log_file.close()
    command = [
        sys.executable,
        "-m",
        "mlblack",
        "catalog",
        "ui",
        "--profile",
        "framework-core",
        "--kind",
        "preset",
        "--port",
        str(port),
        "--headless",
        *extra_args,
    ]
    env = os.environ.copy()
    env.update(dict(env_extra or {}))
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            command,
            cwd=_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            creationflags=creationflags,
        )
    try:
        _wait_for_http_ready(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - cleanup fallback
            proc.kill()
            proc.wait(timeout=10)
        with contextlib.suppress(OSError):
            log_path.unlink()


def _wait_for_sections(page, *, timeout_ms: int = 60000) -> None:
    deadline = time.time() + (timeout_ms / 1000.0)
    while time.time() < deadline:
        if all(page.locator(f"#{section_id}").count() == 1 for section_id in _SECTION_IDS):
            return
        page.wait_for_timeout(500)
    raise AssertionError(f"catalog dashboard did not render expected shared page sections: {_SECTION_IDS}")


def _wait_for_selection(page, selected_key: str, *, timeout_ms: int = 60000) -> None:
    deadline = time.time() + (timeout_ms / 1000.0)
    last_text = ""
    while time.time() < deadline:
        page.wait_for_timeout(1000)
        last_text = page.locator("body").inner_text()
        if selected_key in last_text:
            return
    raise AssertionError(f"catalog dashboard did not reach selection {selected_key!r}.\nLast body text:\n{last_text[:4000]}")


def _click_button_by_text(page, label: str) -> None:
    labels = page.locator("button").all_text_contents()
    try:
        index = labels.index(label)
    except ValueError as exc:
        raise AssertionError(f"catalog dashboard button not found: {label!r}; buttons={labels[:20]}") from exc
    page.locator("button").nth(index).click()


def _results_expander_is_open(page) -> bool:
    details = page.locator('[data-testid="stExpander"]').first.locator("details").first
    return details.get_attribute("open") is not None


def _read_trace_events(trace_path: Path) -> list[dict[str, object]]:
    if not trace_path.exists():
        return []
    lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _wait_for_trace_stable(trace_path: Path, *, minimum_count: int = 0, timeout: float = 30.0) -> list[dict[str, object]]:
    deadline = time.time() + timeout
    last_count = -1
    stable_polls = 0
    last_events: list[dict[str, object]] = []
    while time.time() < deadline:
        events = _read_trace_events(trace_path)
        count = len(events)
        if count >= minimum_count and count == last_count:
            stable_polls += 1
            if stable_polls >= 2:
                return events
        else:
            stable_polls = 0
        last_count = count
        last_events = events
        time.sleep(0.5)
    raise AssertionError(f"catalog trace did not stabilize in time: count={len(last_events)} path={trace_path}")


def _wait_for_trace_delta(trace_path: Path, *, baseline_count: int, timeout: float = 30.0) -> list[dict[str, object]]:
    deadline = time.time() + timeout
    last_count = baseline_count
    stable_polls = 0
    last_events: list[dict[str, object]] = []
    while time.time() < deadline:
        events = _read_trace_events(trace_path)
        count = len(events)
        if count > baseline_count and count == last_count:
            stable_polls += 1
            if stable_polls >= 2:
                return events[baseline_count:]
        else:
            stable_polls = 0
        last_count = count
        last_events = events
        time.sleep(0.5)
    raise AssertionError(
        f"catalog trace did not produce a stable delta after interaction: baseline={baseline_count} current={len(last_events)}"
    )


def test_catalog_ui_shared_page_protocol_sections_e2e():
    with _running_catalog_ui("--query", "mlp_torch", "--column-mode", "full", "--page-size", "25", "--results-collapse", "collapsed") as base_url:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1400})
            page.goto(base_url, wait_until="load", timeout=120000)
            _wait_for_sections(page)
            _wait_for_selection(page, _READY_SELECTED_KEY)

            assert _results_expander_is_open(page) is False
            assert page.locator(f"#{page_protocol.FILTER_SECTION_ID}").inner_text().startswith("FILTER")
            assert page.locator(f"#{page_protocol.RESULT_SECTION_ID}").inner_text().startswith("RESULTS")
            assert page.locator(f"#{page_protocol.DETAIL_SECTION_ID}").inner_text().startswith("DETAIL")
            assert all(page.locator(f"#{row_id}").count() == 1 for row_id in _CONTROL_ROW_IDS)

            browser.close()


def test_catalog_ui_selection_switch_query_trace_e2e():
    trace_file = Path(tempfile.NamedTemporaryFile(prefix="mlblack_catalog_trace_", suffix=".jsonl", delete=False).name)
    try:
        with _running_catalog_ui("--query", "torch", env_extra={shell_protocol.QUERY_TRACE_PATH_ENV: str(trace_file)}) as base_url:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
                page = browser.new_page(viewport={"width": 1600, "height": 1400})
                page.goto(base_url, wait_until="load", timeout=120000)
                _wait_for_sections(page)
                _wait_for_selection(page, _READY_SELECTED_KEY)

                baseline_events = _wait_for_trace_stable(trace_file, minimum_count=1)
                _click_button_by_text(page, "下一项")
                _wait_for_selection(page, _NEXT_SELECTED_KEY)
                delta_events = _wait_for_trace_delta(trace_file, baseline_count=len(baseline_events))

                miss_events = [event for event in delta_events if str(event.get("cache_status")) == "miss"]
                unexpected_miss_events = [
                    event for event in miss_events if str(event.get("loader")) not in {"_load_selected", "_load_neighbors"}
                ]

                assert not unexpected_miss_events, delta_events
                assert {str(event.get("loader")) for event in miss_events} == {"_load_selected", "_load_neighbors"}
                assert len(miss_events) == 2
                assert len(delta_events) <= 8, delta_events

                browser.close()
    finally:
        with contextlib.suppress(OSError):
            trace_file.unlink()
