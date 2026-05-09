from __future__ import annotations

import unittest
from pathlib import Path

from catalog import dashboard as dashboard_module
from catalog import i18n as i18n_module


_KNOWN_MOJIBAKE_FRAGMENTS: tuple[str, ...] = (
    "榛樿",
    "姒傝",
    "鍏崇",
    "鏉ユ簮",
    "涓嶉檺",
    "璇︽儏",
    "鏍囬",
    "鎽樿",
    "褰撳墠绛",
    "瀹舵棌",
    "棰勮",
    "璁粌鍣",
    "杈撳嚭澶",
    "缁勪欢",
    "渚涜兘鍣",
    "鎻掍欢",
    "鏂囨。",
    "绀轰緥",
    "绾挎€",
    "绁炵粡",
)


class TestCatalogChineseUiStrings(unittest.TestCase):
    def test_dashboard_labels_are_clean_chinese(self) -> None:
        self.assertEqual(str(dashboard_module._SORT_LABELS["default"]), "默认排序")
        self.assertEqual(str(dashboard_module._SORT_LABELS["title"]), "标题")
        self.assertEqual(str(dashboard_module._SORT_LABELS["kind"]), "分类")
        self.assertEqual(str(dashboard_module._DETAIL_TAB_LABELS["overview"]), "概览")
        self.assertEqual(str(dashboard_module._DETAIL_TAB_LABELS["relations"]), "关系")
        self.assertEqual(str(dashboard_module._DETAIL_TAB_LABELS["source"]), "来源")

    def test_i18n_helpers_emit_clean_chinese(self) -> None:
        self.assertEqual(i18n_module._kind_label("family"), "家族")
        self.assertEqual(i18n_module._kind_label("trainer"), "训练器")
        self.assertEqual(i18n_module._fallback_title_zh("doc", "doc:catalog", "Catalog DB"), "文档：Catalog DB")
        self.assertEqual(
            i18n_module._fallback_summary_zh("example", "example:run_catalog_dashboard"),
            "已注册的示例条目“run_catalog_dashboard”。",
        )

    def test_i18n_entries_return_clean_bilingual_fields(self) -> None:
        preset = i18n_module.build_entry_i18n_fields(
            kind="preset",
            key="preset:mlp_torch",
            name="mlp_torch",
            summary="",
            metadata={},
        )
        self.assertEqual(str(preset.get("title_zh")), "Torch 多层感知机")
        self.assertIn("神经网络家族", str(preset.get("summary_zh", "")))
        self.assertTrue(tuple(preset.get("use_when_zh", ())))

        trainer = i18n_module.build_entry_i18n_fields(
            kind="trainer",
            key="trainer:mlp_torch",
            name="mlp_torch",
            summary="",
            metadata={},
        )
        self.assertEqual(str(trainer.get("title_zh")), "Torch 多层感知机训练器")
        self.assertIn("兼容 trainer 表面", str(trainer.get("summary_zh", "")))

        doc = i18n_module.build_entry_i18n_fields(
            kind="doc",
            key="doc:catalog_db_protocol",
            name="CATALOG_DB_PROTOCOL",
            summary="",
            metadata={},
        )
        self.assertEqual(str(doc.get("title_zh")), "文档：CATALOG_DB_PROTOCOL")
        self.assertEqual(str(doc.get("summary_zh")), "文档页面：catalog_db_protocol")

        example = i18n_module.build_entry_i18n_fields(
            kind="example",
            key="example:run_catalog_dashboard",
            name="run_catalog_dashboard.py",
            summary="",
            metadata={},
        )
        self.assertEqual(str(example.get("title_zh")), "示例：run_catalog_dashboard.py")
        self.assertEqual(str(example.get("summary_zh")), "示例脚本：run_catalog_dashboard")

    def test_source_files_do_not_contain_known_mojibake_fragments(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative_path in ("catalog/dashboard.py", "catalog/i18n.py"):
            text = (root / relative_path).read_text(encoding="utf-8")
            for fragment in _KNOWN_MOJIBAKE_FRAGMENTS:
                self.assertNotIn(fragment, text, msg=f"{relative_path} still contains mojibake fragment: {fragment}")


if __name__ == "__main__":
    unittest.main()
