"""PubMed search via NCBI E-utilities with parallel fetching and citation counts.

Uses esearch for discovery, efetch for metadata, and elink for PMCIDs + citation counts.
Rate limit: 3 req/sec without API key, 10/sec with NCBI_API_KEY.
"""

import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from urllib.parse import urlencode

import requests

import config

_session = requests.Session()


def _log(msg: str):
    sys.stderr.write(f"[Search] {msg}\n")
    sys.stderr.flush()


def _api_params(extra: dict) -> dict:
    params = dict(extra)
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY
    return params


def _get_xml(url: str, params: dict, timeout: int = 30,
             max_retries: int = 4) -> Optional[ET.Element]:
    backoff = [1, 3, 10, 30]
    for attempt in range(max_retries):
        try:
            resp = _session.get(url, params=_api_params(params), timeout=timeout)
            if resp.status_code == 429:
                delay = backoff[min(attempt, len(backoff) - 1)]
                _log(f"Rate limited (429), retrying in {delay}s "
                     f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return ET.fromstring(resp.content)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                delay = backoff[min(attempt, len(backoff) - 1)]
                _log(f"Rate limited (429), retrying in {delay}s "
                     f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            _log(f"XML fetch failed: {e}")
            return None
        except Exception as e:
            _log(f"XML fetch failed: {e}")
            return None
    _log(f"XML fetch failed after {max_retries} retries (rate limited)")
    return None


def _text(el: Optional[ET.Element], default: str = "") -> str:
    if el is not None and el.text:
        return el.text.strip()
    return default


def _iter_text(el: Optional[ET.Element]) -> str:
    """Get all text content including nested elements."""
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _date_to_pubmed(date_str: str) -> str:
    return date_str.replace("-", "/")


# ── Phase 1: esearch ───────────────────────────────────────────────────

def search_pmids(
    topic: str, from_date: str, to_date: str, max_results: int = 500
) -> list[str]:
    """Search PubMed and return a list of PMIDs."""
    _log(f"Searching for '{topic}' ({from_date} to {to_date}, max={max_results})")

    params = {
        "db": "pubmed",
        "term": topic,
        "retmax": str(max_results),
        "sort": "relevance",
        "datetype": "pdat",
        "mindate": _date_to_pubmed(from_date),
        "maxdate": _date_to_pubmed(to_date),
        "retmode": "xml",
    }

    root = _get_xml(config.ESEARCH_URL, params)
    if root is None:
        _log("esearch request failed")
        return []

    count_el = root.find("Count")
    total = int(_text(count_el, "0"))
    _log(f"Total matching articles in PubMed: {total}")

    id_list = root.find("IdList")
    if id_list is None:
        return []

    pmids = [_text(el) for el in id_list.findall("Id") if el.text]
    _log(f"Retrieved {len(pmids)} PMIDs")
    return pmids


# ── Phase 2: efetch (batched + parallel) ───────────────────────────────

def _parse_single_article(article_el: ET.Element) -> Optional[dict[str, Any]]:
    medline = article_el.find("MedlineCitation")
    if medline is None:
        return None

    pmid = _text(medline.find("PMID"))
    if not pmid:
        return None

    article = medline.find("Article")
    if article is None:
        return None

    title = _iter_text(article.find("ArticleTitle"))

    journal_el = article.find("Journal")
    journal = ""
    if journal_el is not None:
        journal = _text(journal_el.find("Title"))
        if not journal:
            journal = _text(journal_el.find("ISOAbbreviation"))

    authors = []
    author_list_el = article.find("AuthorList")
    if author_list_el is not None:
        for author_el in author_list_el.findall("Author"):
            last = _text(author_el.find("LastName"))
            first = _text(author_el.find("Initials")) or _text(author_el.find("ForeName"))
            if last:
                authors.append(f"{last} {first}".strip() if first else last)

    abstract_el = article.find("Abstract")
    abstract = ""
    if abstract_el is not None:
        parts = []
        for abs_text in abstract_el.findall("AbstractText"):
            label = abs_text.get("Label", "")
            full_text = _iter_text(abs_text)
            if label and label.upper() != "UNLABELLED":
                parts.append(f"{label}: {full_text}")
            else:
                parts.append(full_text)
        abstract = " ".join(parts)

    pub_date = _extract_date(article_el)

    doi = ""
    for eid in article.findall("ELocationID"):
        if eid.get("EIdType") == "doi":
            doi = _text(eid)
            break
    if not doi:
        pubmed_data = article_el.find("PubmedData")
        if pubmed_data is not None:
            for aid in pubmed_data.findall(".//ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = _text(aid)
                    break

    pub_types = []
    pub_type_list = article.find("PublicationTypeList")
    if pub_type_list is not None:
        pub_types = [_text(pt) for pt in pub_type_list.findall("PublicationType") if pt.text]

    mesh_terms = []
    mesh_list = medline.find("MeshHeadingList")
    if mesh_list is not None:
        for mh in mesh_list.findall("MeshHeading"):
            descriptor = mh.find("DescriptorName")
            if descriptor is not None and descriptor.text:
                mesh_terms.append(descriptor.text.strip())

    keywords_list = []
    for kw_list in medline.findall("KeywordList"):
        for kw in kw_list.findall("Keyword"):
            if kw.text:
                keywords_list.append(kw.text.strip())

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "pub_date": pub_date,
        "doi": doi,
        "pub_types": pub_types,
        "mesh_terms": mesh_terms,
        "keywords": keywords_list,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def _extract_date(article_el: ET.Element) -> Optional[str]:
    medline = article_el.find("MedlineCitation")
    if medline is None:
        return None
    article_node = medline.find("Article")
    if article_node is not None:
        for ad in article_node.findall("ArticleDate"):
            d = _parse_date_elements(ad)
            if d:
                return d
    journal = article_node.find("Journal") if article_node is not None else None
    if journal is not None:
        ji = journal.find("JournalIssue")
        if ji is not None:
            pd = ji.find("PubDate")
            if pd is not None:
                d = _parse_date_elements(pd)
                if d:
                    return d
                ml = _text(pd.find("MedlineDate"))
                if ml and len(ml) >= 4:
                    return f"{ml[:4]}-01-01"
    return None


def _parse_date_elements(date_el: ET.Element) -> Optional[str]:
    year = _text(date_el.find("Year"))
    if not year:
        return None
    month = _text(date_el.find("Month"), "01")
    day = _text(date_el.find("Day"), "01")
    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    if month in month_map:
        month = month_map[month]
    try:
        m = int(month)
        d = int(day)
        return f"{year}-{m:02d}-{d:02d}"
    except ValueError:
        return f"{year}-01-01"


def _fetch_batch(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch metadata for a batch of PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    root = _get_xml(config.EFETCH_URL, params, timeout=60)
    if root is None:
        return []

    articles = []
    for article_el in root.findall(".//PubmedArticle"):
        parsed = _parse_single_article(article_el)
        if parsed:
            articles.append(parsed)
    return articles


def fetch_article_metadata(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch metadata for all PMIDs using batched parallel requests."""
    if not pmids:
        return []

    batches = [
        pmids[i : i + config.EFETCH_BATCH_SIZE]
        for i in range(0, len(pmids), config.EFETCH_BATCH_SIZE)
    ]
    _log(f"Fetching metadata in {len(batches)} batch(es)...")

    all_articles = []
    with ThreadPoolExecutor(max_workers=min(config.MAX_WORKERS, len(batches))) as executor:
        futures = {}
        for batch in batches:
            time.sleep(config.RATE_LIMIT_DELAY)
            futures[executor.submit(_fetch_batch, batch)] = len(batch)

        for future in as_completed(futures):
            try:
                articles = future.result()
                all_articles.extend(articles)
            except Exception as e:
                _log(f"Batch fetch error: {e}")

    _log(f"Fetched metadata for {len(all_articles)} articles")
    return all_articles


# ── Phase 3: elink for PMCIDs + citation counts ───────────────────────

def _fetch_links_batch(pmids: list[str]) -> dict[str, dict]:
    """Get PMCIDs and citation counts for a batch of PMIDs."""
    results = {}

    # PMCIDs
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": ",".join(pmids),
        "retmode": "xml",
        "linkname": "pubmed_pmc",
    }
    root = _get_xml(config.ELINK_URL, params, timeout=30)
    if root is not None:
        for linkset in root.findall(".//LinkSet"):
            id_el = linkset.find("IdList/Id")
            if id_el is None:
                continue
            src_pmid = _text(id_el)
            if src_pmid not in results:
                results[src_pmid] = {"pmcid": None, "citation_count": 0}
            link_db = linkset.find("LinkSetDb")
            if link_db is not None:
                link_id = link_db.find("Link/Id")
                if link_id is not None:
                    results[src_pmid]["pmcid"] = f"PMC{_text(link_id)}"

    time.sleep(config.RATE_LIMIT_DELAY)

    # Citation counts via "cited by" links
    params_cite = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "linkname": "pubmed_pubmed_citedin",
    }
    root_cite = _get_xml(config.ELINK_URL, params_cite, timeout=30)
    if root_cite is not None:
        for linkset in root_cite.findall(".//LinkSet"):
            id_el = linkset.find("IdList/Id")
            if id_el is None:
                continue
            src_pmid = _text(id_el)
            if src_pmid not in results:
                results[src_pmid] = {"pmcid": None, "citation_count": 0}
            link_db = linkset.find("LinkSetDb")
            if link_db is not None:
                count = len(link_db.findall("Link"))
                results[src_pmid]["citation_count"] = count

    return results


def enrich_with_links(pmids: list[str]) -> dict[str, dict]:
    """Get PMCIDs and citation counts for all PMIDs (batched, sequential)."""
    if not pmids:
        return {}

    batch_size = 100
    batches = [pmids[i : i + batch_size] for i in range(0, len(pmids), batch_size)]
    n_batches = len(batches)
    _log(f"Fetching links/citations in {n_batches} batch(es) (sequential)...")

    all_results = {}
    for i, batch in enumerate(batches):
        _log(f"elink batch {i + 1}/{n_batches}...")
        try:
            results = _fetch_links_batch(batch)
            all_results.update(results)
        except Exception as e:
            _log(f"Link fetch error on batch {i + 1}: {e}")
        time.sleep(config.RATE_LIMIT_DELAY)

    pmc_count = sum(1 for v in all_results.values() if v.get("pmcid"))
    _log(f"Found {pmc_count} articles with PMC full text, enriched citation counts")
    return all_results
