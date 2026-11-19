"""
darglobal_scraper.py
--------------------
Dynamic crawler for DarGlobal (darglobal.co.uk).
Discovers real project and navigation links automatically from the homepage
and /projects, then scrapes verified pages.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "darglobal"
DELAY = 1.5  # seconds between requests

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

IGNORE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".pdf",
    ".css", ".js", ".ico", ".woff", ".woff2", ".ttf"
)


def url_to_slug(url: str) -> str:
    path = re.sub(r"https?://[^/]+", "", url).strip("/")
    if not path:
        return "home"
    return re.sub(r"[^a-zA-Z0-9_-]", "_", path)[:60]


def extract_data(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    desc_tag = soup.find("meta", attrs={"name": "description"})
    description = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""

    headings = []
    for level in ["h1", "h2", "h3"]:
        for tag in soup.find_all(level):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) > 3:
                headings.append({"level": level, "text": text})

    paragraphs = []
    for p in soup.find_all(["p", "div", "li"]):
        text = p.get_text(separator=" ", strip=True)
        if 25 < len(text) < 1500 and text not in paragraphs:
            if not any(text in h["text"] for h in headings):
                paragraphs.append(text)

    return {
        "url": url,
        "title": title,
        "description": description,
        "headings": headings,
        "text": paragraphs,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def save_txt(record: dict, slug: str) -> str:
    path = OUTPUT_DIR / f"{slug}.txt"
    lines = [
        f"URL: {record['url']}",
        f"Title: {record['title']}",
        f"Description: {record['description']}",
        "",
        "=== HEADINGS ===",
    ]
    for h in record["headings"]:
        lines.append(f"[{h['level'].upper()}] {h['text']}")
    lines += ["", "=== TEXT ==="]
    for para in record["text"]:
        lines.append(para)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(path)


def save_pdf(record: dict, slug: str) -> None:
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        title = record.get("title", slug).encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 8, f"DarGlobal: {title}")
        pdf.ln(3)

        pdf.set_font("Helvetica", "I", 9)
        url = record.get("url", "").encode("latin-1", "replace").decode("latin-1")
        pdf.cell(0, 6, f"URL: {url}", ln=True)
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 10)
        for para in record.get("text", [])[:35]:
            clean_para = para.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 5, clean_para)
            pdf.ln(2)

        pdf_path = OUTPUT_DIR / f"{slug}.pdf"
        pdf.output(str(pdf_path))
    except Exception:
        pass


def discover_links(session: cffi_requests.Session, seed_urls: list[str]) -> list[str]:
    """Dynamically discover real internal links from DarGlobal pages."""
    found_links = set(seed_urls)
    domain = "darglobal.co.uk"

    for url in seed_urls:
        try:
            resp = session.get(url, headers=HEADERS, timeout=25)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)
                if domain in parsed.netloc and not parsed.path.lower().endswith(IGNORE_EXTENSIONS):
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                    found_links.add(clean_url)
        except Exception:
            pass
        time.sleep(0.5)

    return sorted(list(found_links))


# Seed URLs covering DarGlobal corporate, masterplans, and signature luxury developments
SEED_PROJECT_URLS = [
    "https://darglobal.co.uk",
    "https://darglobal.co.uk/projects",
    "https://darglobal.co.uk/urban-oasis",
    "https://darglobal.co.uk/urban-oasis-by-missoni",
    "https://darglobal.co.uk/projects/urban-oasis",
    "https://darglobal.co.uk/dg1",
    "https://darglobal.co.uk/projects/dg1",
    "https://darglobal.co.uk/da-vinci-tower",
    "https://darglobal.co.uk/projects/da-vinci-tower",
    "https://darglobal.co.uk/the-astera",
    "https://darglobal.co.uk/w-residences-dubai-downtown",
    "https://darglobal.co.uk/tierra-viva",
    "https://darglobal.co.uk/marea",
    "https://darglobal.co.uk/les-vagues",
    "https://darglobal.co.uk/trump-international-hotel-tower-dubai",
    "https://darglobal.co.uk/aida",
    "https://darglobal.co.uk/aida-trump-international-hotel",
    "https://darglobal.co.uk/aida-trump-villas",
    "https://darglobal.co.uk/aida-trump-international-cliff-villas",
    "https://darglobal.co.uk/aida-coastal-investment-villas",
    "https://darglobal.co.uk/Sunrise-Haven-Luxury-Villas",
    "https://darglobal.co.uk/amour-sans-detour",
    "https://darglobal.co.uk/about",
]


def main(max_pages: int = 30):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = cffi_requests.Session(impersonate="chrome120")

    print("Discovering live links from DarGlobal dynamically …")
    discovered = discover_links(session, SEED_PROJECT_URLS)
    
    # Deduplicate and build prioritized crawl list
    crawl_list = []
    for u in SEED_PROJECT_URLS:
        if u not in crawl_list and len(crawl_list) < max_pages:
            crawl_list.append(u)
    for u in discovered:
        if u not in crawl_list and len(crawl_list) < max_pages:
            crawl_list.append(u)
    
    print(f"Found {len(discovered)} live URLs. Crawling up to {len(crawl_list)} verified pages:")
    all_records = []
    total_chars = 0

    for url in crawl_list:
        slug = url_to_slug(url)
        print(f"Fetching: {url}  (slug={slug})")
        try:
            resp = session.get(url, headers=HEADERS, timeout=25)
            if resp.status_code != 200:
                print(f"  Skipped (HTTP {resp.status_code})")
                continue
            record = extract_data(url, resp.text)
            page_chars = sum(len(t) for t in record["text"]) + len(record["title"])
            total_chars += page_chars
            save_txt(record, slug)
            save_pdf(record, slug)
            all_records.append(record)
            print(f"  OK — {len(record['headings'])} headings, {len(record['text'])} text blocks, {page_chars} chars (txt + pdf saved)")
        except Exception as exc:
            print(f"  Notice: {exc}")

        time.sleep(DELAY)

    json_path = OUTPUT_DIR / "darglobal_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n=== DarGlobal scrape complete ===")
    print(f"Pages crawled successfully: {len(all_records)}")
    print(f"Total characters collected: {total_chars:,}")


if __name__ == "__main__":
    main()
