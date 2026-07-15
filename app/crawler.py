"""
Website crawler for the college site.

Responsibilities:
  * Discover every internal HTML page (BFS crawl, across the main domain
    AND its subdomains, e.g. www.example.ac.in, exam.example.ac.in,
    hostel.example.ac.in all count as "internal").
  * Seed the crawl from sitemap.xml (if present) in addition to BFS,
    to catch pages that aren't linked from the homepage nav.
  * Download every linked document (PDF, DOCX, DOC, TXT, etc.) to disk,
    detecting real file type from content bytes, not just URL/headers.
  * Look for links/documents inside <a>, <area>, <iframe>, <embed>,
    and <object> tags -- not just <a href>.
  * Save raw HTML pages to disk for later parsing.
  * Retry transient failures before giving up on a URL.
  * Be polite: respect a delay between requests.

Usage:
    python -m app.crawler
"""
from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from app.config import settings

# File extensions we treat as downloadable documents (not HTML).
DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt", ".pptx", ".ppt", ".xls", ".xlsx"}
SKIP_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".css", ".js", ".woff", ".woff2", ".ttf", ".mp4", ".mp3", ".zip", ".rar"}

# Second-level suffixes where the "root domain" needs 3 labels instead of 2
# (e.g. nitsikkim.ac.in -> root is "nitsikkim.ac.in", not "ac.in").
THREE_LABEL_SUFFIXES = {"ac.in", "co.in", "gov.in", "edu.in", "org.in", "net.in", "res.in"}

# Tags/attributes we scan for links, beyond plain <a href>.
LINK_ATTR_TAGS = [
    ("a", "href"),
    ("area", "href"),
    ("iframe", "src"),
    ("embed", "src"),
    ("object", "data"),
]

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class Crawler:
    def __init__(self) -> None:
        self.base_url: str = settings.site_url.rstrip("/")
        self.base_netloc = urlparse(self.base_url).netloc
        self.root_domain = self._root_domain(self.base_netloc)
        # Optional explicit allow-list from config, e.g.
        # settings.extra_allowed_domains = ["some-external-partner-portal.edu"]
        self.extra_allowed_domains = set(getattr(settings, "extra_allowed_domains", []) or [])
        self.raw_dir = settings.raw_path
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.crawl_user_agent})
        self.visited: set[str] = set()
        self.queued: set[str] = set()
        self.documents: list[dict] = []  # metadata for downloaded files
        self.pages: list[dict] = []      # metadata for HTML pages
        self.skipped: list[dict] = []    # metadata for things we couldn't classify

        # None or 0 in config means "no cap" -- crawl until queue is empty.
        self.max_pages = getattr(settings, "crawl_max_pages", None) or float("inf")

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------
    def _root_domain(self, netloc: str) -> str:
        """
        Reduce a netloc to its registrable root domain so subdomains
        compare equal, e.g.:
          www.nitsikkim.ac.in     -> nitsikkim.ac.in
          exam.nitsikkim.ac.in    -> nitsikkim.ac.in
          hostel.nitsikkim.ac.in  -> nitsikkim.ac.in
        """
        host = netloc.split(":")[0].lower()
        parts = host.split(".")
        if len(parts) <= 2:
            return host
        last_two = ".".join(parts[-2:])
        if last_two in THREE_LABEL_SUFFIXES and len(parts) >= 3:
            return ".".join(parts[-3:])
        return last_two

    def _normalize(self, url: str) -> str:
        """Strip fragments and normalize trailing slashes for dedup."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        query = parsed.query
        rebuilt = f"{parsed.scheme}://{parsed.netloc}{path}"
        if query:
            rebuilt += f"?{query}"
        return rebuilt

    def _is_internal(self, url: str) -> bool:
        netloc = urlparse(url).netloc
        if not netloc:
            return False
        if netloc in self.extra_allowed_domains:
            return True
        return self._root_domain(netloc) == self.root_domain

    def _url_ext(self, url: str) -> str:
        return Path(urlparse(url).path).suffix.lower()

    def _should_skip(self, url: str) -> bool:
        return self._url_ext(url) in SKIP_EXTS

    def _safe_filename(self, url: str, forced_ext: str | None = None) -> str:
        """Create a deterministic, filesystem-safe filename from a URL."""
        parsed = urlparse(url)
        suffix = forced_ext or (Path(parsed.path).suffix.lower() or ".html")
        digest = hashlib.sha256(url.encode()).hexdigest()[:12]
        name = Path(parsed.path).stem or "index"
        name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]
        return f"{name}_{digest}{suffix}"

    # ------------------------------------------------------------------
    # Content-type sniffing (fixes PHP-served PDFs being saved as HTML)
    # ------------------------------------------------------------------
    def _sniff_kind(self, content: bytes, content_type: str, url: str) -> str:
        head = content[:8]

        if head.startswith(b"%PDF"):
            return "pdf"
        if head.startswith(b"\xd0\xcf\x11\xe0"):
            return "doc_ole"
        if head.startswith(b"PK\x03\x04"):
            return "zip_based"

        url_ext = self._url_ext(url)
        
        # We must check for HTML content BEFORE trusting the URL extension.
        # This prevents 404 error pages or login walls from being saved as PDFs.
        if "text/html" in content_type or b"<html" in content[:1000].lower():
            return "html"
            
        if url_ext in DOCUMENT_EXTS:
            return "other_doc"
            
        if "application/pdf" in content_type:
            return "pdf"

        return "unknown"

    def _ext_for_kind(self, kind: str, url: str) -> str:
        if kind == "pdf":
            return ".pdf"
        if kind == "doc_ole":
            url_ext = self._url_ext(url)
            return url_ext if url_ext in {".doc", ".xls", ".ppt"} else ".doc"
        if kind == "zip_based":
            url_ext = self._url_ext(url)
            return url_ext if url_ext in {".docx", ".xlsx", ".pptx"} else ".docx"
        if kind == "other_doc":
            return self._url_ext(url) or ".dat"
        return ".html"

    # ------------------------------------------------------------------
    # Networking with retries
    # ------------------------------------------------------------------
    def _fetch(self, url: str) -> requests.Response | None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.session.get(url, timeout=30, allow_redirects=True)
            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    tqdm.write(f"[WARN] Failed to fetch {url} after {MAX_RETRIES} attempts: {exc}")
                    return None
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        return None

    # ------------------------------------------------------------------
    # Sitemap discovery (seeds queue with pages that may not be linked
    # from the homepage/nav at all)
    # ------------------------------------------------------------------
    def _discover_sitemap_urls(self) -> list[str]:
        candidates = [
            urljoin(self.base_url + "/", "sitemap.xml"),
            urljoin(self.base_url + "/", "sitemap_index.xml"),
        ]
        found: list[str] = []
        for sitemap_url in candidates:
            resp = self._fetch(sitemap_url)
            if resp is None or resp.status_code != 200:
                continue
            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError:
                continue
            # Handles both <urlset><url><loc> and <sitemapindex><sitemap><loc>
            for loc in root.iter():
                if loc.tag.lower().endswith("loc") and loc.text:
                    found.append(loc.text.strip())
        if found:
            tqdm.write(f"🗺️  Found {len(found)} URLs via sitemap")
        return found

    # ------------------------------------------------------------------
    # Core crawl loop
    # ------------------------------------------------------------------
    def crawl(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        queue: deque[str] = deque([self.base_url])
        self.queued.add(self.base_url)

        if getattr(settings, "extra_seed_urls", ""):
            for url in settings.extra_seed_urls.split(","):
                url = url.strip()
                if url:
                    norm = self._normalize(url)
                    if norm not in self.queued:
                        self.queued.add(norm)
                        queue.append(norm)

        for sitemap_url in self._discover_sitemap_urls():
            norm = self._normalize(sitemap_url)
            if self._is_internal(norm) and norm not in self.queued:
                self.queued.add(norm)
                queue.append(norm)

        pbar = tqdm(desc="Crawling", unit="page")
        while queue and len(self.visited) < self.max_pages:
            url = queue.popleft()
            url = self._normalize(url)
            if url in self.visited or not self._is_internal(url):
                continue

            resp = self._fetch(url)
            if resp is None:
                continue

            self.visited.add(url)
            pbar.update(1)

            content_type = resp.headers.get("Content-Type", "").lower()

            if self._should_skip(url):
                time.sleep(settings.crawl_delay)
                continue

            kind = self._sniff_kind(resp.content, content_type, url)

            if kind in ("pdf", "doc_ole", "zip_based", "other_doc"):
                self._save_document(url, resp.content, kind)
                time.sleep(settings.crawl_delay)
                continue

            if kind == "html":
                self._save_html(url, resp.text)
                self._extract_links(url, resp.text, queue)
                time.sleep(settings.crawl_delay)
                continue

            tqdm.write(f"[SKIP] Could not classify content at {url} (content-type={content_type!r})")
            self.skipped.append({"url": url, "content_type": content_type})
            time.sleep(settings.crawl_delay)

        pbar.close()
        self._write_manifest()
        print(
            f"\n✅ Crawl complete: {len(self.pages)} HTML pages, "
            f"{len(self.documents)} documents downloaded, "
            f"{len(self.skipped)} unclassified/skipped."
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _save_html(self, url: str, html: str) -> None:
        fname = self._safe_filename(url, forced_ext=".html")
        out_path = self.raw_dir / fname
        out_path.write_text(html, encoding="utf-8", errors="ignore")
        self.pages.append({"url": url, "file": str(out_path), "type": "html"})

    def _save_document(self, url: str, content: bytes, kind: str) -> None:
        ext = self._ext_for_kind(kind, url)
        fname = self._safe_filename(url, forced_ext=ext)
        out_path = self.raw_dir / fname
        out_path.write_bytes(content)
        self.documents.append({
            "url": url,
            "file": str(out_path),
            "type": ext.lstrip("."),
            "sniffed_kind": kind,
        })
        tqdm.write(f"  📄 Downloaded document ({kind}): {url}")

    # ------------------------------------------------------------------
    # Link extraction
    # ------------------------------------------------------------------
    def _extract_links(self, page_url: str, html: str, queue: deque[str]) -> None:
        soup = BeautifulSoup(html, "lxml")

        for tag_name, attr in LINK_ATTR_TAGS:
            for tag in soup.find_all(tag_name):
                raw = tag.get(attr)
                if not raw:
                    continue
                raw = raw.strip()
                if not raw or raw.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                absolute = urljoin(page_url, raw)
                absolute = self._normalize(absolute)
                if self._should_skip(absolute):
                    continue
                if not self._is_internal(absolute):
                    continue
                if absolute in self.visited or absolute in self.queued:
                    continue
                self.queued.add(absolute)
                queue.append(absolute)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def _write_manifest(self) -> None:
        import json

        manifest = {
            "site": self.base_url,
            "root_domain": self.root_domain,
            "pages_visited": len(self.visited),
            "html_pages": self.pages,
            "documents": self.documents,
            "skipped": self.skipped,
        }
        manifest_path = self.raw_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"📝 Manifest written to {manifest_path}")


def main() -> None:
    Crawler().crawl()


if __name__ == "__main__":
    main()