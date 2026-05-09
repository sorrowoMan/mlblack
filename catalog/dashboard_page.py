from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Mapping, Sequence

PAGE_TOP_ANCHOR_ID = "catalog-page-top"
HERO_SECTION_ID = "catalog-section-hero"
STATS_SECTION_ID = "catalog-section-stats"
FILTER_SECTION_ID = "catalog-section-filter"
RESULT_SECTION_ID = "catalog-section-results"
DETAIL_SECTION_ID = "catalog-section-detail"
PRIMARY_CONTROLS_ROW_ID = "catalog-controls-primary"
SECONDARY_CONTROLS_ROW_ID = "catalog-controls-secondary"

PAGE_PROTOCOL_STYLE = """
.catalog-shell-block {
    margin-top: 0.5rem;
    margin-bottom: 0.4rem;
}
.catalog-shell-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.catalog-shell-copy {
    min-width: 0;
}
.catalog-shell-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
    color: #8b6840;
    margin-bottom: 0.12rem;
}
.catalog-shell-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #2f2115;
    margin-bottom: 0.06rem;
}
.catalog-shell-subtitle {
    font-size: 0.88rem;
    color: #65513a;
}
.catalog-shell-note {
    font-size: 0.83rem;
    color: #816847;
    text-align: right;
    white-space: nowrap;
    padding-top: 0.16rem;
}
"""


@dataclass(frozen=True)
class HeroSpec:
    icon_text: str
    kicker: str
    title: str
    subtitle: str


@dataclass(frozen=True)
class StatCardSpec:
    label: str
    value: str
    note: str


@dataclass(frozen=True)
class SectionHeaderSpec:
    section_id: str
    label: str
    title: str
    subtitle: str = ""
    note: str = ""


@dataclass(frozen=True)
class ControlSlotSpec:
    slot_id: str
    width: float
    label: str
    help: str = ""
    caption: str = ""
    placeholder: str = ""


@dataclass(frozen=True)
class ControlRowSpec:
    row_id: str
    section_id: str
    slots: tuple[ControlSlotSpec, ...]


@dataclass
class ControlSlotHandle:
    spec: ControlSlotSpec
    column: Any

    def caption(self, text: str | None = None) -> None:
        note = self.spec.caption if text is None else text
        if str(note or "").strip():
            self.column.caption(str(note))

    @property
    def label(self) -> str:
        return str(self.spec.label)

    @property
    def help(self) -> str:
        return str(self.spec.help)

    @property
    def placeholder(self) -> str:
        return str(self.spec.placeholder)


def render_top_anchor(st: Any, *, anchor_id: str = PAGE_TOP_ANCHOR_ID) -> None:
    st.markdown(
        f"<div id='{escape(str(anchor_id))}' data-catalog-anchor='page-top'></div>",
        unsafe_allow_html=True,
    )


def render_hero(st: Any, spec: HeroSpec) -> None:
    st.markdown(
        (
            f"<div id='{HERO_SECTION_ID}' data-catalog-section='hero' class='catalog-hero'>"
            "<div class='catalog-hero-head'>"
            "<div class='catalog-brand'>"
            f"<div class='catalog-icon'>{escape(spec.icon_text)}</div>"
            "<div>"
            f"<div class='catalog-kicker'>{escape(spec.kicker)}</div>"
            f"<div class='catalog-title'>{escape(spec.title)}</div>"
            "</div></div></div>"
            f"<div class='catalog-sub'>{escape(spec.subtitle)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_stat_cards(
    st: Any,
    cards: Sequence[StatCardSpec],
    *,
    column_weights: Sequence[float] | None = None,
) -> None:
    if not cards:
        return
    weights = tuple(column_weights or (1.0,) * len(cards))
    cols = st.columns(weights if len(weights) == len(cards) else (1.0,) * len(cards))
    st.markdown(
        f"<div id='{STATS_SECTION_ID}' data-catalog-section='stats'></div>",
        unsafe_allow_html=True,
    )
    for col, card in zip(cols, cards):
        col.markdown(
            (
                "<div class='catalog-stat'>"
                f"<div class='catalog-stat-label'>{escape(card.label)}</div>"
                f"<div class='catalog-stat-value'>{escape(card.value)}</div>"
                f"<div class='catalog-stat-note'>{escape(card.note)}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_section_header(st: Any, spec: SectionHeaderSpec) -> None:
    subtitle_html = (
        f"<div class='catalog-shell-subtitle'>{escape(spec.subtitle)}</div>"
        if str(spec.subtitle or "").strip()
        else ""
    )
    note_html = (
        f"<div class='catalog-shell-note'>{escape(spec.note)}</div>"
        if str(spec.note or "").strip()
        else ""
    )
    st.markdown(
        (
            f"<div id='{escape(spec.section_id)}' data-catalog-section='{escape(spec.label.lower())}' class='catalog-shell-block'>"
            "<div class='catalog-shell-header'>"
            "<div class='catalog-shell-copy'>"
            f"<div class='catalog-shell-label'>{escape(spec.label)}</div>"
            f"<div class='catalog-shell-title'>{escape(spec.title)}</div>"
            f"{subtitle_html}"
            "</div>"
            f"{note_html}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_control_row(st: Any, spec: ControlRowSpec) -> Mapping[str, ControlSlotHandle]:
    st.markdown(
        (
            f"<div id='{escape(spec.row_id)}' "
            f"data-catalog-control-row='{escape(spec.section_id)}' "
            "class='catalog-shell-block'></div>"
        ),
        unsafe_allow_html=True,
    )
    columns = st.columns(tuple(slot.width for slot in spec.slots))
    return {
        str(slot.slot_id): ControlSlotHandle(spec=slot, column=column)
        for slot, column in zip(spec.slots, columns)
    }
