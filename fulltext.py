"""Full-text and PDF downloader for PubMed articles.

Downloads free full text from PMC (via efetch or OA service) and
falls back to trafilatura for publisher pages accessible via DOI.
Supports HTTP and FTP protocols for PDF downloads.
"""

import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftplib import FTP
from typing import Optional
from urllib.parse import urlparse
from io import BytesIO

import requests
import trafilatura

import config
from database import Database

_session = requests.Session()
_session.headers.update({
    "User-Agent": "PubMedAgent/1.0 (research literature aggregator)"
})


def _log(msg: str):
    sys.stderr.write(f"[FullText] {msg}\n")
    sys.stderr.flush()


def _fetch_pmc_fulltext(pmcid: str) -> Optional[str]:
    """Fetch full text XML from PMC and extract plain text."""
    pmc_id_num = pmcid.replace("PMC", "")
    params = {
        "db": "pmc",
        "id": pmc_id_num,
        "retmode": "xml",
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY

    try:
        resp = _session.get(config.EFETCH_URL, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        text_parts = []
        for body in root.iter("body"):
            for sec in body.iter("sec"):
                title_el = sec.find("title")
                if title_el is not None and title_el.text:
                    text_parts.append(f"\n## {title_el.text.strip()}\n")
                for p in sec.findall("p"):
                    p_text = "".join(p.itertext()).strip()
                    if p_text:
                        text_parts.append(p_text)

        if not text_parts:
            for body in root.iter("body"):
                all_text = "".join(body.itertext()).strip()
                if all_text:
                    text_parts.append(all_text)

        return "\n\n".join(text_parts) if text_parts else None
    except Exception as e:
        _log(f"PMC fetch failed for {pmcid}: {e}")
        return None


def _fetch_pmc_pdf_url(pmcid: str) -> Optional[str]:
    """Get PDF download URL from PMC OA service."""
    params = {"id": pmcid, "format": "pdf"}
    try:
        resp = _session.get(config.PMC_OA_URL, params=params, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for record in root.findall(".//record"):
            link = record.find("link")
            if link is not None:
                href = link.get("href", "")
                fmt = link.get("format", "")
                if fmt == "pdf" or href.endswith(".pdf"):
                    return href
        return None
    except Exception:
        return None


def _download_pdf_ftp(url: str) -> Optional[bytes]:
    """Download a PDF file via FTP."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc
        path = parsed.path
        
        ftp = FTP(host, timeout=60)
        ftp.login()
        
        buf = BytesIO()
        ftp.retrbinary(f"RETR {path}", buf.write)
        ftp.quit()
        
        data = buf.getvalue()
        if len(data) < 1000:
            return None
        return data
    except Exception as e:
        _log(f"FTP PDF download failed from {url}: {e}")
        return None


def _download_pdf(url: str) -> Optional[bytes]:
    """Download a PDF file via HTTP or FTP."""
    if url.startswith("ftp://"):
        return _download_pdf_ftp(url)
    
    try:
        resp = _session.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and not url.endswith(".pdf"):
            return None
        data = resp.content
        if len(data) < 1000:
            return None
        return data
    except Exception as e:
        _log(f"PDF download failed from {url}: {e}")
        return None


def _fetch_via_doi(doi: str) -> Optional[str]:
    """Fetch article text via DOI using trafilatura."""
    if not doi:
        return None
    url = f"https://doi.org/{doi}"
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text) > 200:
                return text
    except Exception as e:
        _log(f"DOI fetch failed for {doi}: {e}")
    return None


def _process_article(article: dict, db_path: str) -> dict:
    """Download full text and PDF for a single article. Returns status dict."""
    thread_db = Database(db_path)
    thread_db.connect()

    pmid = article["pmid"]
    pmcid = article.get("pmcid")
    doi = article.get("doi")
    article_id = article["id"]
    result = {"pmid": pmid, "fulltext": False, "pdf": False}

    text_content = None
    source = None

    if pmcid:
        text_content = _fetch_pmc_fulltext(pmcid)
        if text_content:
            source = "pmc"
        time.sleep(config.RATE_LIMIT_DELAY)

    if not text_content and doi:
        text_content = _fetch_via_doi(doi)
        if text_content:
            source = "publisher"

    if text_content and source:
        thread_db.insert_fulltext(article_id, pmid, text_content, source)
        result["fulltext"] = True

    if pmcid:
        pdf_url = _fetch_pmc_pdf_url(pmcid)
        if pdf_url:
            time.sleep(config.RATE_LIMIT_DELAY)
            pdf_data = _download_pdf(pdf_url)
            if pdf_data:
                thread_db.insert_pdf(article_id, pmid, pdf_data, pdf_url)
                result["pdf"] = True

    thread_db.close()
    return result


def download_fulltexts(
    db: Database,
    search_id: int,
    max_workers: int = 4,
) -> dict:
    """Download full texts and PDFs for all eligible articles in a search.

    Uses a conservative worker count for NCBI rate limits.
    """
    articles = db.get_articles_needing_fulltext(search_id)
    if not articles:
        _log("No articles need full text download")
        return {"downloaded": 0, "pdfs": 0, "total": 0}

    _log(f"Downloading full texts for {len(articles)} articles...")
    downloaded = 0
    pdfs = 0

    effective_workers = min(max_workers, 3 if not config.NCBI_API_KEY else 6)
    db_path = db.db_path

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        for article in articles:
            futures[executor.submit(_process_article, article, db_path)] = article["pmid"]

        for future in as_completed(futures):
            pmid = futures[future]
            try:
                result = future.result()
                if result["fulltext"]:
                    downloaded += 1
                if result["pdf"]:
                    pdfs += 1
                if (downloaded + pdfs) % 10 == 0:
                    _log(f"Progress: {downloaded} texts, {pdfs} PDFs downloaded")
            except Exception as e:
                _log(f"Error processing {pmid}: {e}")

    _log(f"Done: {downloaded} full texts, {pdfs} PDFs from {len(articles)} articles")
    return {"downloaded": downloaded, "pdfs": pdfs, "total": len(articles)}
