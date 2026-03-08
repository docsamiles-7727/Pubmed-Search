"""SQLite storage for PubMed research articles, full texts, and summaries.

Optimized for 512GB RAM with aggressive caching, WAL mode, and memory-mapped I/O.
"""

import json
import sqlite3
import sys
from datetime import datetime
from typing import Any, Optional

import config


def _log(msg: str):
    sys.stderr.write(f"[DB] {msg}\n")
    sys.stderr.flush()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    from_date TEXT,
    to_date TEXT,
    max_results INTEGER,
    total_found INTEGER DEFAULT 0,
    articles_fetched INTEGER DEFAULT 0,
    fulltexts_downloaded INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id INTEGER NOT NULL REFERENCES searches(id),
    pmid TEXT NOT NULL,
    pmcid TEXT,
    doi TEXT,
    title TEXT,
    abstract TEXT,
    authors TEXT,
    journal TEXT,
    pub_date TEXT,
    pub_types TEXT,
    mesh_terms TEXT,
    keywords TEXT,
    citation_count INTEGER DEFAULT 0,
    has_free_fulltext INTEGER DEFAULT 0,
    fulltext_url TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(search_id, pmid)
);

CREATE TABLE IF NOT EXISTS full_texts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    pmid TEXT NOT NULL,
    text_content TEXT,
    source TEXT,
    downloaded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pdfs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    pmid TEXT NOT NULL,
    pdf_data BLOB,
    pdf_url TEXT,
    file_size INTEGER,
    downloaded_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id INTEGER NOT NULL REFERENCES searches(id),
    topic TEXT,
    model TEXT,
    summary_md TEXT,
    docx_path TEXT,
    md_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS intermediate_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id INTEGER NOT NULL REFERENCES searches(id),
    step_type TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    date_range_start TEXT,
    date_range_end TEXT,
    articles_count INTEGER,
    summary_text TEXT,
    model TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(search_id, step_type, step_index)
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_articles_search ON articles(search_id);
CREATE INDEX IF NOT EXISTS idx_articles_pmid ON articles(pmid);
CREATE INDEX IF NOT EXISTS idx_articles_pmcid ON articles(pmcid);
CREATE INDEX IF NOT EXISTS idx_articles_citations ON articles(citation_count DESC);
CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(pub_date DESC);
CREATE INDEX IF NOT EXISTS idx_fulltexts_article ON full_texts(article_id);
CREATE INDEX IF NOT EXISTS idx_fulltexts_pmid ON full_texts(pmid);
CREATE INDEX IF NOT EXISTS idx_pdfs_article ON pdfs(article_id);
CREATE INDEX IF NOT EXISTS idx_summaries_search ON summaries(search_id);
CREATE INDEX IF NOT EXISTS idx_intermediates_search ON intermediate_summaries(search_id);
"""


class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(config.DB_PATH)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self.conn is not None:
            return self.conn
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = -2000000")
        self.conn.execute("PRAGMA mmap_size = 8589934592")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA foreign_keys = ON")
        _log(f"Connected to {self.db_path}")
        return self.conn

    def init_schema(self):
        conn = self.connect()
        conn.executescript(_SCHEMA)
        conn.commit()
        _log("Schema initialized")

    def create_indexes(self):
        conn = self.connect()
        conn.executescript(_INDEXES)
        conn.commit()
        _log("Indexes created")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    # ── Searches ────────────────────────────────────────────────────────

    def create_search(
        self, topic: str, from_date: str, to_date: str, max_results: int
    ) -> int:
        conn = self.connect()
        cursor = conn.execute(
            "INSERT INTO searches (topic, from_date, to_date, max_results) VALUES (?, ?, ?, ?)",
            (topic, from_date, to_date, max_results),
        )
        conn.commit()
        return cursor.lastrowid

    def update_search(self, search_id: int, **kwargs):
        conn = self.connect()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        conn.execute(
            f"UPDATE searches SET {sets} WHERE id = ?",
            (*kwargs.values(), search_id),
        )
        conn.commit()

    def get_search(self, search_id: int) -> Optional[dict]:
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM searches WHERE id = ?", (search_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_searches(self, limit: int = 20) -> list[dict]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM searches ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Articles ────────────────────────────────────────────────────────

    def bulk_insert_articles(
        self, search_id: int, articles: list[dict[str, Any]]
    ) -> int:
        if not articles:
            return 0
        conn = self.connect()
        rows = []
        for a in articles:
            rows.append((
                search_id,
                a.get("pmid", ""),
                a.get("pmcid"),
                a.get("doi"),
                a.get("title"),
                a.get("abstract"),
                json.dumps(a["authors"]) if isinstance(a.get("authors"), list) else a.get("authors", ""),
                a.get("journal"),
                a.get("pub_date"),
                json.dumps(a["pub_types"]) if isinstance(a.get("pub_types"), list) else a.get("pub_types"),
                json.dumps(a["mesh_terms"]) if isinstance(a.get("mesh_terms"), list) else a.get("mesh_terms"),
                json.dumps(a["keywords"]) if isinstance(a.get("keywords"), list) else a.get("keywords"),
                a.get("citation_count", 0),
                1 if a.get("has_free_fulltext") else 0,
                a.get("fulltext_url"),
            ))

        inserted = 0
        for i in range(0, len(rows), config.BULK_INSERT_BATCH_SIZE):
            batch = rows[i : i + config.BULK_INSERT_BATCH_SIZE]
            conn.executemany(
                """INSERT OR IGNORE INTO articles
                   (search_id, pmid, pmcid, doi, title, abstract, authors, journal,
                    pub_date, pub_types, mesh_terms, keywords, citation_count,
                    has_free_fulltext, fulltext_url)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch,
            )
            inserted += len(batch)
        conn.commit()
        _log(f"Inserted {inserted} articles for search {search_id}")
        return inserted

    def get_articles(
        self,
        search_id: int,
        has_fulltext_only: bool = False,
        order_by: str = "citation_count DESC",
    ) -> list[dict]:
        conn = self.connect()
        where = "WHERE a.search_id = ?"
        params: list[Any] = [search_id]
        if has_fulltext_only:
            where += " AND a.has_free_fulltext = 1"
        rows = conn.execute(
            f"""SELECT a.*, ft.text_content
                FROM articles a
                LEFT JOIN full_texts ft ON ft.article_id = a.id
                {where}
                ORDER BY {order_by}""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_articles_needing_fulltext(self, search_id: int) -> list[dict]:
        conn = self.connect()
        rows = conn.execute(
            """SELECT a.* FROM articles a
               LEFT JOIN full_texts ft ON ft.article_id = a.id
               WHERE a.search_id = ? AND a.has_free_fulltext = 1 AND ft.id IS NULL
               ORDER BY a.id""",
            (search_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_article(self, article_id: int, **kwargs):
        conn = self.connect()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        conn.execute(
            f"UPDATE articles SET {sets} WHERE id = ?",
            (*kwargs.values(), article_id),
        )
        conn.commit()

    def update_articles_bulk(self, updates: list[tuple]):
        """Updates: list of (pmcid, citation_count, has_free_fulltext, pmid, search_id)."""
        if not updates:
            return
        conn = self.connect()
        conn.executemany(
            """UPDATE articles SET pmcid = ?, citation_count = ?, has_free_fulltext = ?
               WHERE pmid = ? AND search_id = ?""",
            updates,
        )
        conn.commit()

    # ── Full Texts ──────────────────────────────────────────────────────

    def insert_fulltext(
        self, article_id: int, pmid: str, text_content: str, source: str
    ):
        conn = self.connect()
        conn.execute(
            "INSERT INTO full_texts (article_id, pmid, text_content, source) VALUES (?, ?, ?, ?)",
            (article_id, pmid, text_content, source),
        )
        conn.commit()

    def bulk_insert_fulltexts(self, records: list[tuple]):
        """Records: list of (article_id, pmid, text_content, source)."""
        if not records:
            return
        conn = self.connect()
        conn.executemany(
            "INSERT INTO full_texts (article_id, pmid, text_content, source) VALUES (?, ?, ?, ?)",
            records,
        )
        conn.commit()
        _log(f"Inserted {len(records)} full texts")

    # ── PDFs ────────────────────────────────────────────────────────────

    def insert_pdf(
        self, article_id: int, pmid: str, pdf_data: bytes, pdf_url: str
    ):
        conn = self.connect()
        conn.execute(
            "INSERT INTO pdfs (article_id, pmid, pdf_data, pdf_url, file_size) VALUES (?, ?, ?, ?, ?)",
            (article_id, pmid, pdf_data, pdf_url, len(pdf_data)),
        )
        conn.commit()

    # ── Summaries ───────────────────────────────────────────────────────

    def insert_summary(
        self,
        search_id: int,
        topic: str,
        model: str,
        summary_md: str,
        docx_path: str,
        md_path: str,
    ) -> int:
        conn = self.connect()
        cursor = conn.execute(
            """INSERT INTO summaries (search_id, topic, model, summary_md, docx_path, md_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (search_id, topic, model, summary_md, docx_path, md_path),
        )
        conn.commit()
        return cursor.lastrowid

    def get_summary(self, search_id: int) -> Optional[dict]:
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM summaries WHERE search_id = ? ORDER BY created_at DESC LIMIT 1",
            (search_id,),
        ).fetchone()
        return dict(row) if row else None

    # ── Intermediate Summaries ──────────────────────────────────────────

    def insert_intermediate(
        self,
        search_id: int,
        step_type: str,
        step_index: int,
        summary_text: str,
        model: str,
        date_range_start: str = "",
        date_range_end: str = "",
        articles_count: int = 0,
    ) -> int:
        conn = self.connect()
        cursor = conn.execute(
            """INSERT OR REPLACE INTO intermediate_summaries
               (search_id, step_type, step_index, summary_text, model,
                date_range_start, date_range_end, articles_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (search_id, step_type, step_index, summary_text, model,
             date_range_start, date_range_end, articles_count),
        )
        conn.commit()
        return cursor.lastrowid

    def get_intermediate(
        self, search_id: int, step_type: str, step_index: int
    ) -> Optional[dict]:
        conn = self.connect()
        row = conn.execute(
            """SELECT * FROM intermediate_summaries
               WHERE search_id = ? AND step_type = ? AND step_index = ?""",
            (search_id, step_type, step_index),
        ).fetchone()
        return dict(row) if row else None

    def get_all_intermediates(
        self, search_id: int, step_type: Optional[str] = None
    ) -> list[dict]:
        conn = self.connect()
        if step_type:
            rows = conn.execute(
                """SELECT * FROM intermediate_summaries
                   WHERE search_id = ? AND step_type = ?
                   ORDER BY step_index""",
                (search_id, step_type),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM intermediate_summaries
                   WHERE search_id = ?
                   ORDER BY step_type, step_index""",
                (search_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_intermediates(self, search_id: int):
        conn = self.connect()
        conn.execute(
            "DELETE FROM intermediate_summaries WHERE search_id = ?",
            (search_id,),
        )
        conn.commit()

    # ── Stats ───────────────────────────────────────────────────────────

    def get_search_stats(self, search_id: int) -> dict:
        conn = self.connect()
        article_count = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE search_id = ?", (search_id,)
        ).fetchone()[0]
        fulltext_count = conn.execute(
            """SELECT COUNT(*) FROM full_texts ft
               JOIN articles a ON ft.article_id = a.id
               WHERE a.search_id = ?""",
            (search_id,),
        ).fetchone()[0]
        pdf_count = conn.execute(
            """SELECT COUNT(*) FROM pdfs p
               JOIN articles a ON p.article_id = a.id
               WHERE a.search_id = ?""",
            (search_id,),
        ).fetchone()[0]
        return {
            "articles": article_count,
            "full_texts": fulltext_count,
            "pdfs": pdf_count,
        }
