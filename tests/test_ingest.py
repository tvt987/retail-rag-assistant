from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

from app.ingest import (IngestReport, clean_text, drop_shared_boilerplate, html_to_text, load_documents,
                        markdown_to_text, parse_file)

FIXTURES = Path(__file__).parent / "fixtures"


class ParsingTests(unittest.TestCase):
    def test_markdown_syntax_is_stripped_but_text_kept(self):
        md = "# Returns\n\n- Keep the **receipt** for [14 days](https://example.com).\n\n> Opened `hygiene` items excluded."
        text = markdown_to_text(md)
        self.assertNotIn("#", text)
        self.assertNotIn("**", text)
        self.assertNotIn("https://", text)
        self.assertIn("Keep the receipt for 14 days.", text)
        self.assertIn("Opened hygiene items excluded.", text)

    def test_html_drops_scripts_styles_and_navigation(self):
        html = ("<html><head><style>p{}</style></head><body><nav>Home | Contact</nav>"
                "<h1>Pickup</h1><p>Ready in 2 hours.</p><script>track()</script></body></html>")
        text = clean_text(html_to_text(html))
        self.assertEqual(text, "Pickup\n\nReady in 2 hours.")

    @unittest.skipUnless(find_spec("pypdf"), "pypdf not installed")
    def test_pdf_text_is_extracted_and_hyphenation_repaired(self):
        text = clean_text(parse_file(FIXTURES / "gift_card_policy.pdf"))
        self.assertIn("valid for 12 months from the date of purchase", text)


class PreprocessingTests(unittest.TestCase):
    def test_clean_text_normalises_unicode_whitespace_and_line_break_hyphens(self):
        raw = "Re-\nfunds   take 5 to 10 days.\r\n\r\n\r\nFULLWIDTH１４ days\x07"
        self.assertEqual(clean_text(raw), "Refunds take 5 to 10 days.\n\nFULLWIDTH14 days")

    def test_paragraphs_shared_across_documents_are_removed_as_boilerplate(self):
        footer = "Contact support at any time."
        docs = {"a.md": f"Alpha policy.\n\n{footer}", "b.md": f"Beta policy.\n\n{footer}\n\nBeta policy."}
        cleaned = drop_shared_boilerplate(docs)
        self.assertEqual(cleaned, {"a.md": "Alpha policy.", "b.md": "Beta policy."})


class LoadingTests(unittest.TestCase):
    def test_load_reports_skipped_empty_and_duplicate_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# A\n\nShipping takes 3 days.", encoding="utf-8")
            (root / "copy.txt").write_text("A\n\nShipping takes 3 days.", encoding="utf-8")
            (root / "empty.txt").write_text("   \n", encoding="utf-8")
            (root / "notes.docx").write_bytes(b"ignored: unsupported extension")
            report = IngestReport()
            docs = load_documents(root, report)
        self.assertEqual(list(docs), ["a.md"])
        self.assertEqual(report.duplicates, ["copy.txt == a.md"])
        self.assertEqual(report.skipped, {"empty.txt": "no extractable text"})

    def test_repository_documents_parse_into_clean_paragraphs(self):
        docs = load_documents(Path(__file__).resolve().parents[1] / "data")
        self.assertEqual(sorted(docs), ["returns.md", "shipping.md", "store_pickup.html"])
        self.assertNotIn("<", docs["store_pickup.html"])
        self.assertNotIn("Home | Policies", docs["store_pickup.html"])
        self.assertTrue(docs["returns.md"].startswith("Returns policy (fictional demo)\n\n"))


if __name__ == "__main__":
    unittest.main()
