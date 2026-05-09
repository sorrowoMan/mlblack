from __future__ import annotations

import unittest

from catalog import list_entries


class TestCatalogProfile(unittest.TestCase):
    def test_framework_core_excludes_doc_and_example(self) -> None:
        core_entries = list_entries(profile="framework-core")
        leaked = [e for e in core_entries if e.kind in {"doc", "example"}]
        self.assertEqual(leaked, [])

    def test_default_includes_non_core_entries(self) -> None:
        default_entries = list_entries(profile="default")
        kinds = {e.kind for e in default_entries}
        self.assertIn("doc", kinds)
        self.assertIn("example", kinds)


if __name__ == "__main__":
    unittest.main()
