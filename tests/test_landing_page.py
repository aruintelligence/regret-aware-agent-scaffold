import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "docs"


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.stylesheets = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.stylesheets.append(attrs.get("href"))


class LandingPageTests(unittest.TestCase):
    def test_static_site_is_self_contained_and_has_valid_internal_links(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        parser = LinkCollector()
        parser.feed(html)
        self.assertIn("styles.css", parser.stylesheets)
        self.assertTrue((ROOT / "styles.css").is_file())
        for href in parser.links:
            if href.startswith("#"):
                self.assertIn(href[1:], parser.ids)
            elif not href.startswith(("http://", "https://", "mailto:")):
                self.assertTrue((ROOT / href).is_file(), href)

    def test_real_evidence_is_presented_with_limitations(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        for expected in ("78.5%", "82.1%", "+3.6 pp", "2,000",
                         "research", "not", "Victoria", "pilot"):
            self.assertIn(expected.lower(), html.lower())


if __name__ == "__main__":
    unittest.main()
