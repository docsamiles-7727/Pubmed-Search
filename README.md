# PubMed Literature Search and Summarization Agent

Searches PubMed via NCBI E-utilities, downloads free full-text articles and PDFs from PMC, stores everything in a memory-optimized SQLite database, and generates structured medical literature reviews using your choice of LLM provider.

## Supported LLM Providers

| Provider | Default Model | Flag |
|----------|--------------|------|
| **xAI (Grok)** | `grok-4-0709` | `--provider xai` (default) |
| **Google (Gemini)** | `gemini-2.5-flash` | `--provider google` |
| **Anthropic (Claude)** | `claude-sonnet-4-20250514` | `--provider anthropic` |

## Setup

```bash
# Activate the Python environment
pyenv

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Full Pipeline (search + download + summarize)

```bash
# Fetch ALL available articles (up to 10,000 max)
python pubmed_agent.py pipeline \
    --topic "pancreatic cancer immunotherapy" \
    --from-date 2020-01-01 \
    --to-date 2026-03-07 \
    --all

# Limit to specific number (default: 200)
python pubmed_agent.py pipeline \
    --topic "pancreatic cancer immunotherapy" \
    --from-date 2020-01-01 \
    --to-date 2026-03-07 \
    --max-results 500

# Use Google Gemini with all articles
python pubmed_agent.py pipeline \
    --topic "pancreatic cancer immunotherapy" \
    --from-date 2020-01-01 \
    --to-date 2026-03-07 \
    --all \
    --provider google

# Use a specific model
python pubmed_agent.py pipeline \
    --topic "pancreatic cancer immunotherapy" \
    --from-date 2020-01-01 \
    --to-date 2026-03-07 \
    --all \
    --provider xai --model grok-3
```

### Individual Steps

```bash
# Search PubMed and store metadata (fetch all)
python pubmed_agent.py search --topic "BRCA mutations breast cancer" \
    --from-date 2022-01-01 --to-date 2026-03-07 --all

# Download full texts and PDFs for an existing search
python pubmed_agent.py download --search-id 1

# Summarize from database (with provider override)
python pubmed_agent.py summarize --search-id 1 --provider anthropic

# Show all searches
python pubmed_agent.py status
```

## Output

The agent generates two output files in `output/{topic}_{timestamp}/`:

- `literature_review.md` -- Markdown format
- `literature_review.docx` -- Microsoft Word format

Both include structured sections covering pathobiology, molecular characteristics, treatment approaches, experimental therapies, study limitations, and comprehensive citations in Vancouver/NLM style.

## Architecture

- **search.py** -- NCBI E-utilities (esearch/efetch/elink) with parallel batched fetching
- **fulltext.py** -- PMC full-text XML download + trafilatura DOI fallback + PDF download
- **database.py** -- SQLite with WAL mode, 2GB page cache, 8GB mmap for 512GB RAM systems
- **summarizer.py** -- Multi-provider LLM summarization (xAI, Google, Anthropic) with multi-pass chunking
- **output.py** -- Markdown and Word document generation with formatted citations
- **prompts.py** -- Structured prompt templates for medical literature review

## API Keys

- **XAI_API_KEY** -- xAI/Grok (default provider)
- **GOOGLE_API_KEY** -- Google Gemini (alternative)
- **ANTHROPIC_API_KEY** -- Anthropic Claude (alternative)
- **NCBI_API_KEY** -- Optional but recommended (increases rate limit from 3 to 10 req/sec)
