# Pubmed Search - Contents Overview

**Directory:** `/Users/stevenmiles/Cursor/Pubmed Search`  
**Generated:** 2026-03-22 16:27:16

---

## Git Repository Status

- **Is Git Repository:** Yes
- **Remote URL:** `https://github.com/docsamiles-7727/Pubmed-Search.git`
- **Last Commit:** Chronological left-fold summarization, local LLM support, rate limit fixes 2026-03-08 11:54:48 (-0700)
- **Commit Hash:** `7dfdf21b`

---

## Directory Structure

### Directories (4)

- `.git`
- `.specstory`
- `__pycache__`
- `output`

### Files ({len(analysis['files'])})

#### Scripts (.py files)
- `config.py`
- `database.py`
- `fulltext.py`
- `output.py`
- `prompts.py`
- `pubmed_agent.py`
- `search.py`
- `summarizer.py`

#### Databases
- `pubmed_research.db`

#### Configuration Files
- `.env`
- `README.md`
- `requirements.txt`

#### Other Files
- `.DS_Store`
- `.cursorindexingignore`
- `.env.example`
- `.gitignore`
- `.python-project`
- `contents.md`


---

## Script Descriptions

### `config.py`

**Path:** `/Users/stevenmiles/Cursor/Pubmed Search/config.py`

No docstring found

---

### `database.py`

**Path:** `/Users/stevenmiles/Cursor/Pubmed Search/database.py`

Optimized for 512GB RAM with aggressive caching, WAL mode, and memory-mapped I/O.

---

### `prompts.py`

**Path:** `/Users/stevenmiles/Cursor/Pubmed Search/prompts.py`



---

### `search.py`

**Path:** `/Users/stevenmiles/Cursor/Pubmed Search/search.py`

Uses esearch for discovery, efetch for metadata, and elink for PMCIDs + citation counts.

Rate limit: 3 req/sec without API key, 10/sec with NCBI_API_KEY.

---

### `summarizer.py`

**Path:** `/Users/stevenmiles/Cursor/Pubmed Search/summarizer.py`

Supports cloud APIs (xAI Grok, Google Gemini, Anthropic Claude) and local

inference servers (Ollama, LM Studio, Inferencer) via OpenAI-compatible endpoints.

Articles are sorted oldest-first, chunked, summarized, then iteratively merged

so that newer findings take precedence. All intermediates persist in SQLite for

crash resilience and resumability.

---

### `pubmed_agent.py`

**Path:** `/Users/stevenmiles/Cursor/Pubmed Search/pubmed_agent.py`

Searches PubMed, downloads free full-text articles and PDFs,

stores everything in SQLite, and generates structured literature

reviews using cloud LLMs (xAI, Google, Anthropic) or local inference

servers (Ollama, LM Studio, Inferencer).

---

### `output.py`

**Path:** `/Users/stevenmiles/Cursor/Pubmed Search/output.py`

Converts the LLM-generated summary markdown into professionally formatted documents.

---

### `fulltext.py`

**Path:** `/Users/stevenmiles/Cursor/Pubmed Search/fulltext.py`

Downloads free full text from PMC (via efetch or OA service) and

falls back to trafilatura for publisher pages accessible via DOI.

Supports HTTP and FTP protocols for PDF downloads.

---

---

## Overall Assessment

- **Total Files:** 18
- **Total Directories:** 4
- **Scripts Found:** 8
- **Databases Found:** 1
- **Backup Locations:** 0

**Purpose:** Database storage and data persistence
