from __future__ import annotations
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "DataAnalystAgent/1.0 (+https://example.local)"
}

def _wiki_url(page: str) -> str:
    # Allow either slug or title
    slug = page.replace(" ", "_")
    if slug.lower().startswith("http://") or slug.lower().startswith("https://"):
        return slug
    return f"https://en.wikipedia.org/wiki/{slug}"

def scrape_wikipedia_table(page: str, table_index: int = 0, columns: Optional[List[str]] = None, top_k: int = 50) -> pd.DataFrame:
    """
    Scrape a 'wikitable' from a Wikipedia page using BeautifulSoup.
    Returns a pandas DataFrame. Optionally select columns and head(top_k).
    """
    url = _wiki_url(page)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = soup.select("table.wikitable")
    if not tables:
        raise ValueError("No wikitable found on the page.")
    if table_index >= len(tables):
        raise ValueError(f"table_index {table_index} out of range; found {len(tables)} tables.")

    table = tables[table_index]
    df = pd.read_html(str(table))[0]

    if columns:
        existing = [c for c in columns if c in df.columns]
        if existing:
            df = df[existing]

    if top_k and top_k > 0:
        df = df.head(top_k)

    return df
