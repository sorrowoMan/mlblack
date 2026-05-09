from __future__ import annotations

from my_project.etf_quant_interval_proxy.config import EtfQuantIntervalConfig
from my_project.etf_quant_interval_proxy.orchestration import run_suite


def run_etf_quant_interval_proxy(
    cfg: EtfQuantIntervalConfig | None = None,
    *,
    suite_id: str,
):
    return run_suite(cfg or EtfQuantIntervalConfig(), suite_id=suite_id)


__all__ = ["EtfQuantIntervalConfig", "run_etf_quant_interval_proxy"]
