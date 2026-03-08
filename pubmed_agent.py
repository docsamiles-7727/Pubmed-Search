#!/usr/bin/env python3
"""PubMed Literature Search and Summarization Agent.

Searches PubMed, downloads free full-text articles and PDFs,
stores everything in SQLite, and generates structured literature
reviews using xAI Grok, Google Gemini, or Anthropic Claude.
"""

import argparse
import sys
import time
from datetime import datetime

import config
from database import Database
from search import search_pmids, fetch_article_metadata, enrich_with_links
from fulltext import download_fulltexts
from summarizer import run_summarization
from output import generate_outputs


def _log(msg: str):
    sys.stderr.write(f"\n{'='*60}\n[Agent] {msg}\n{'='*60}\n")
    sys.stderr.flush()


def cmd_search(args, db: Database):
    """Search PubMed and store article metadata."""
    max_results = 10000 if getattr(args, "all", False) else args.max_results
    
    _log(f"Searching PubMed: '{args.topic}' ({args.from_date} to {args.to_date})")
    if getattr(args, "all", False):
        print("Fetching ALL available articles (up to 10,000 max)")

    search_id = db.create_search(
        args.topic, args.from_date, args.to_date, max_results
    )
    print(f"Search ID: {search_id}")

    pmids = search_pmids(args.topic, args.from_date, args.to_date, max_results)
    if not pmids:
        db.update_search(search_id, status="completed", total_found=0)
        print("No articles found.")
        return search_id

    db.update_search(search_id, total_found=len(pmids), status="fetching")
    print(f"Found {len(pmids)} PMIDs, fetching metadata...")

    articles = fetch_article_metadata(pmids)
    if not articles:
        db.update_search(search_id, status="completed", articles_fetched=0)
        print("Failed to fetch article metadata.")
        return search_id

    inserted = db.bulk_insert_articles(search_id, articles)
    print(f"Stored {inserted} articles in database")

    print("Enriching with PMC links and citation counts...")
    link_data = enrich_with_links(pmids)

    updates = []
    for pmid, info in link_data.items():
        updates.append((
            info.get("pmcid"),
            info.get("citation_count", 0),
            1 if info.get("pmcid") else 0,
            pmid,
            search_id,
        ))
    db.update_articles_bulk(updates)

    pmc_count = sum(1 for v in link_data.values() if v.get("pmcid"))
    print(f"Found {pmc_count} articles with free full text in PMC")

    db.create_indexes()
    db.update_search(
        search_id,
        status="searched",
        articles_fetched=inserted,
    )
    print(f"Search complete. Search ID: {search_id}")
    return search_id


def cmd_download(args, db: Database):
    """Download full texts and PDFs for a search."""
    search_id = args.search_id
    search = db.get_search(search_id)
    if not search:
        print(f"Search {search_id} not found.")
        return

    _log(f"Downloading full texts for search {search_id}: '{search['topic']}'")
    db.update_search(search_id, status="downloading")

    result = download_fulltexts(db, search_id, max_workers=args.workers)

    db.update_search(
        search_id,
        status="downloaded",
        fulltexts_downloaded=result["downloaded"],
    )

    print(f"Downloaded {result['downloaded']} full texts, {result['pdfs']} PDFs "
          f"from {result['total']} eligible articles")


def cmd_summarize(args, db: Database):
    """Generate a literature review summary."""
    search_id = args.search_id
    search = db.get_search(search_id)
    if not search:
        print(f"Search {search_id} not found.")
        return

    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    effective_provider = provider or config.LLM_PROVIDER
    effective_model = model or config.LLM_MODEL or config.get_default_model(effective_provider)

    _log(f"Summarizing search {search_id}: '{search['topic']}' with {effective_provider}/{effective_model}")
    db.update_search(search_id, status="summarizing")

    summary_md = run_summarization(db, search_id, provider=provider, model=model)

    paths = generate_outputs(summary_md, search["topic"])

    db.insert_summary(
        search_id=search_id,
        topic=search["topic"],
        model=f"{effective_provider}/{effective_model}",
        summary_md=summary_md,
        docx_path=str(paths["docx_path"]),
        md_path=str(paths["md_path"]),
    )

    db.update_search(search_id, status="completed", finished_at=datetime.now().isoformat())

    print(f"\nLiterature review generated ({effective_provider}/{effective_model}):")
    print(f"  Markdown: {paths['md_path']}")
    print(f"  Word:     {paths['docx_path']}")
    print(f"  Dir:      {paths['output_dir']}")


def cmd_pipeline(args, db: Database):
    """Full pipeline: search -> download -> summarize."""
    t0 = time.time()

    search_id = cmd_search(args, db)
    if not search_id:
        return

    stats = db.get_search_stats(search_id)
    if stats["articles"] == 0:
        print("No articles to process.")
        return

    args.search_id = search_id
    args.workers = getattr(args, "workers", 4)
    cmd_download(args, db)

    cmd_summarize(args, db)

    elapsed = time.time() - t0
    final_stats = db.get_search_stats(search_id)
    print(f"\nPipeline complete in {elapsed:.1f}s")
    print(f"  Articles: {final_stats['articles']}")
    print(f"  Full texts: {final_stats['full_texts']}")
    print(f"  PDFs: {final_stats['pdfs']}")


def cmd_status(args, db: Database):
    """Show status of searches."""
    searches = db.list_searches()
    if not searches:
        print("No searches found.")
        return

    print(f"{'ID':>4}  {'Status':<12}  {'Articles':>8}  {'Topic'}")
    print("-" * 70)
    for s in searches:
        print(f"{s['id']:>4}  {s['status']:<12}  {s['articles_fetched'] or 0:>8}  {s['topic'][:40]}")


def main():
    parser = argparse.ArgumentParser(
        description="PubMed Literature Search and Summarization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with ALL available articles (default: xai/grok-4-0709)
  python pubmed_agent.py pipeline --topic "pancreatic cancer immunotherapy" \\
      --from-date 2020-01-01 --to-date 2026-03-07 --all

  # Full pipeline with specific limit
  python pubmed_agent.py pipeline --topic "pancreatic cancer immunotherapy" \\
      --from-date 2020-01-01 --to-date 2026-03-07 --max-results 500

  # Use Google Gemini instead
  python pubmed_agent.py pipeline --topic "BRCA mutations" \\
      --from-date 2022-01-01 --to-date 2026-03-07 --all --provider google

  # Use Anthropic Claude with specific model
  python pubmed_agent.py summarize --search-id 1 \\
      --provider anthropic --model claude-sonnet-4-20250514

  # Search only (fetch all articles)
  python pubmed_agent.py search --topic "BRCA mutations breast cancer" \\
      --from-date 2022-01-01 --to-date 2026-03-07 --all

  # Download full texts for an existing search
  python pubmed_agent.py download --search-id 1

  # Summarize from database
  python pubmed_agent.py summarize --search-id 1

  # Show all searches
  python pubmed_agent.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    llm_help = "LLM provider: xai, google, or anthropic (default: xai)"
    model_help = "LLM model override (default: provider-specific, e.g. grok-4-0709)"

    # Pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Full pipeline: search + download + summarize")
    p_pipe.add_argument("--topic", required=True, help="Search topic")
    p_pipe.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    p_pipe.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    p_pipe.add_argument("--max-results", type=int, default=200, help="Max articles to retrieve (ignored if --all is set)")
    p_pipe.add_argument("--all", action="store_true", help="Fetch ALL available articles (up to 10,000 max)")
    p_pipe.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    p_pipe.add_argument("--provider", choices=["xai", "google", "anthropic"], default=None, help=llm_help)
    p_pipe.add_argument("--model", default=None, help=model_help)

    # Search
    p_search = subparsers.add_parser("search", help="Search PubMed and store metadata")
    p_search.add_argument("--topic", required=True, help="Search topic")
    p_search.add_argument("--from-date", required=True, help="Start date (YYYY-MM-DD)")
    p_search.add_argument("--to-date", required=True, help="End date (YYYY-MM-DD)")
    p_search.add_argument("--max-results", type=int, default=200, help="Max articles to retrieve (ignored if --all is set)")
    p_search.add_argument("--all", action="store_true", help="Fetch ALL available articles (up to 10,000 max)")

    # Download
    p_dl = subparsers.add_parser("download", help="Download full texts for an existing search")
    p_dl.add_argument("--search-id", type=int, required=True, help="Search ID")
    p_dl.add_argument("--workers", type=int, default=4, help="Parallel download workers")

    # Summarize
    p_sum = subparsers.add_parser("summarize", help="Generate literature review from DB")
    p_sum.add_argument("--search-id", type=int, required=True, help="Search ID")
    p_sum.add_argument("--provider", choices=["xai", "google", "anthropic"], default=None, help=llm_help)
    p_sum.add_argument("--model", default=None, help=model_help)

    # Status
    subparsers.add_parser("status", help="Show status of all searches")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    db = Database()
    db.init_schema()

    try:
        if args.command == "pipeline":
            cmd_pipeline(args, db)
        elif args.command == "search":
            cmd_search(args, db)
        elif args.command == "download":
            cmd_download(args, db)
        elif args.command == "summarize":
            cmd_summarize(args, db)
        elif args.command == "status":
            cmd_status(args, db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
