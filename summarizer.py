"""Multi-provider LLM summarization for PubMed literature reviews.

Supports xAI (Grok), Google (Gemini), and Anthropic (Claude) via direct HTTP APIs.
Default: xAI grok-4-1-fast-reasoning.
"""

import sys
import time

import requests

import config
from database import Database
from prompts import CHUNK_SUMMARY_PROMPT, SYNTHESIS_PROMPT, format_articles_for_prompt

_session = requests.Session()


def _log(msg: str):
    sys.stderr.write(f"[Summarizer] {msg}\n")
    sys.stderr.flush()


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _call_xai(prompt: str, api_key: str, model: str, max_tokens: int, temperature: float) -> str:
    resp = _session.post(
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_google(prompt: str, api_key: str, model: str, max_tokens: int, temperature: float) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    resp = _session.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                return part["text"]
    return ""


def _call_anthropic(prompt: str, api_key: str, model: str, max_tokens: int, temperature: float) -> str:
    resp = _session.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


_PROVIDERS = {
    "xai": _call_xai,
    "google": _call_google,
    "anthropic": _call_anthropic,
}


def _call_llm(prompt: str, provider: str, model: str, max_tokens: int = 8192, temperature: float = 0.3) -> str:
    api_key = config.get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"API key not set for provider '{provider}'. Check your .env file.")

    call_fn = _PROVIDERS.get(provider)
    if not call_fn:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {', '.join(_PROVIDERS)}")

    _log(f"Calling {provider}/{model} (~{_estimate_tokens(prompt)} input tokens)")
    return call_fn(prompt, api_key, model, max_tokens, temperature)


def _chunk_articles(articles: list[dict], max_tokens: int = 200_000) -> list[list[dict]]:
    chunks = []
    current_chunk = []
    current_tokens = 0

    for article in articles:
        article_text = format_articles_for_prompt([article])
        article_tokens = _estimate_tokens(article_text)

        if current_tokens + article_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(article)
        current_tokens += article_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def run_summarization(
    db: Database,
    search_id: int,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Full summarization pipeline: chunk -> summarize -> synthesize.

    Returns the final markdown summary.
    """
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_MODEL
    if not model:
        model = config.get_default_model(provider)

    search = db.get_search(search_id)
    if not search:
        raise ValueError(f"Search {search_id} not found")

    topic = search["topic"]
    from_date = search["from_date"]
    to_date = search["to_date"]

    articles = db.get_articles(search_id, order_by="citation_count DESC")
    if not articles:
        raise ValueError(f"No articles found for search {search_id}")

    stats = db.get_search_stats(search_id)
    _log(f"Summarizing {len(articles)} articles ({stats['full_texts']} with full text)")
    _log(f"Provider: {provider}, Model: {model}")

    chunks = _chunk_articles(articles)
    _log(f"Split into {len(chunks)} chunk(s)")

    if len(chunks) == 1:
        articles_text = format_articles_for_prompt(chunks[0])
        prompt = SYNTHESIS_PROMPT.format(
            topic=topic,
            from_date=from_date,
            to_date=to_date,
            total_articles=len(articles),
            fulltext_count=stats["full_texts"],
            batch_summaries=f"### Single Batch Analysis\n\n{articles_text}",
        )
        _log(f"Single-pass summarization (~{_estimate_tokens(prompt)} tokens)...")
        summary_md = _call_llm(prompt, provider, model, max_tokens=16384, temperature=0.2)
    else:
        batch_summaries = []
        for i, chunk in enumerate(chunks):
            _log(f"Processing chunk {i+1}/{len(chunks)}...")
            articles_text = format_articles_for_prompt(chunk)
            prompt = CHUNK_SUMMARY_PROMPT.format(articles_text=articles_text)
            summary = _call_llm(prompt, provider, model, max_tokens=8192, temperature=0.3)
            batch_summaries.append(summary)
            if i < len(chunks) - 1:
                time.sleep(2)

        combined = "\n\n---\n\n".join(
            f"### Batch {i+1}\n{s}" for i, s in enumerate(batch_summaries)
        )
        prompt = SYNTHESIS_PROMPT.format(
            topic=topic,
            from_date=from_date,
            to_date=to_date,
            total_articles=len(articles),
            fulltext_count=stats["full_texts"],
            batch_summaries=combined,
        )
        _log(f"Synthesizing final review (~{_estimate_tokens(prompt)} tokens)...")
        summary_md = _call_llm(prompt, provider, model, max_tokens=16384, temperature=0.2)

    _log("Summarization complete")
    return summary_md
