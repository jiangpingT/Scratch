from __future__ import annotations

from bs4 import BeautifulSoup  # type: ignore
from readability import Document  # type: ignore


def extract_main_text(html: str, fallback_length: int = 6000) -> str:
    try:
        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "html.parser")
        text = " ".join(soup.get_text(" ").split())
        if len(text) > 200:
            return text[:fallback_length]
    except Exception:
        pass
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:fallback_length]

