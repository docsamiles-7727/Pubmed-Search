"""Multi-provider LLM summarization using chronological left-fold merge.

Supports cloud APIs (xAI Grok, Google Gemini, Anthropic Claude) and local
inference servers (Ollama, LM Studio, Inferencer) via OpenAI-compatible endpoints.
Articles are sorted oldest-first, chunked, summarized, then iteratively merged
so that newer findings take precedence. All intermediates persist in SQLite for
crash resilience and resumability.
"""

import re
import sys
import time

import requests

import config
from database import Database
from prompts import (
    CHUNK_SUMMARY_PROMPT,
    MERGE_PROMPT,
    SYNTHESIS_PROMPT,
    format_articles_for_prompt,
)

_session = requests.Session()

_BACKOFF_DELAYS = [5, 20, 60]
_MAX_RETRIES = 3


def _log(msg: str):
    sys.stderr.write(f"[Summarizer] {msg}\n")
    sys.stderr.flush()


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).strip()


def _timeout_for_tokens(token_count: int) -> int:
    """Scale HTTP timeout with prompt size: at least 300s, +1s per 200 tokens."""
    return max(300, token_count // 200)


def _call_xai(prompt: str, api_key: str, model: str, max_tokens: int,
              temperature: float, timeout: int) -> str:
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
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_google(prompt: str, api_key: str, model: str, max_tokens: int,
                 temperature: float, timeout: int) -> str:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={api_key}")
    resp = _session.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                return part["text"]
    return ""


def _call_anthropic(prompt: str, api_key: str, model: str, max_tokens: int,
                    temperature: float, timeout: int) -> str:
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
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def _call_local(prompt: str, api_key: str, model: str, max_tokens: int,
                temperature: float, timeout: int,
                api_url: str = "") -> str:
    """OpenAI-compatible endpoint used by Ollama, LM Studio, and Inferencer."""
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "local":
        headers["Authorization"] = f"Bearer {api_key}"

    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens

    resp = _session.post(api_url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    text = ""
    if "choices" in data:
        text = data["choices"][0]["message"]["content"]
    elif "message" in data and "content" in data["message"]:
        text = data["message"]["content"]
    elif "response" in data:
        text = data["response"]
    elif "content" in data:
        if isinstance(data["content"], list):
            for block in data["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    break
        elif isinstance(data["content"], str):
            text = data["content"]
    else:
        _log(f"Unexpected response format from {api_url}. "
             f"Keys: {list(data.keys())}")
        _log(f"Full response: {str(data)[:500]}")
        raise RuntimeError(
            f"Could not parse response from local server. "
            f"Response keys: {list(data.keys())}")

    return _strip_thinking(text)


_PROVIDERS = {
    "xai": _call_xai,
    "google": _call_google,
    "anthropic": _call_anthropic,
    "ollama": _call_local,
    "lmstudio": _call_local,
    "inferencer": _call_local,
}


def _call_llm(prompt: str, provider: str, model: str,
              max_tokens: int = 8192, temperature: float = 0.3) -> str:
    """Call LLM with retry and exponential backoff on timeout / 5xx errors."""
    is_local = provider in config.LOCAL_PROVIDERS
    api_key = config.get_api_key(provider)
    if not api_key and not is_local:
        raise RuntimeError(
            f"API key not set for provider '{provider}'. Check your .env file.")

    call_fn = _PROVIDERS.get(provider)
    if not call_fn:
        raise ValueError(
            f"Unknown provider: {provider}. Choose from: {', '.join(_PROVIDERS)}")

    input_tokens = _estimate_tokens(prompt)
    timeout = _timeout_for_tokens(input_tokens)
    if is_local:
        timeout = max(timeout, 600)
    _log(f"Calling {provider}/{model} (~{input_tokens} input tokens, "
         f"timeout={timeout}s)")

    extra_kwargs = {}
    if is_local:
        extra_kwargs["api_url"] = config.get_api_url(provider)

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            return call_fn(prompt, api_key, model, max_tokens, temperature,
                           timeout, **extra_kwargs)
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            last_error = e
            delay = _BACKOFF_DELAYS[min(attempt, len(_BACKOFF_DELAYS) - 1)]
            _log(f"Timeout/connection error (attempt {attempt + 1}/{_MAX_RETRIES}), "
                 f"retrying in {delay}s: {e}")
            time.sleep(delay)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code >= 500:
                last_error = e
                delay = _BACKOFF_DELAYS[min(attempt, len(_BACKOFF_DELAYS) - 1)]
                _log(f"Server error {e.response.status_code} "
                     f"(attempt {attempt + 1}/{_MAX_RETRIES}), "
                     f"retrying in {delay}s")
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(
        f"LLM call failed after {_MAX_RETRIES} attempts: {last_error}")


def _chunk_articles(articles: list[dict],
                    max_tokens: int | None = None) -> list[list[dict]]:
    if max_tokens is None:
        max_tokens = config.MAX_CHUNK_TOKENS
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
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


def _date_range_for_chunk(chunk: list[dict]) -> tuple[str, str]:
    """Return (earliest pub_date, latest pub_date) from a chunk."""
    dates = [a.get("pub_date", "") for a in chunk if a.get("pub_date")]
    if not dates:
        return ("", "")
    return (min(dates), max(dates))


def run_summarization(
    db: Database,
    search_id: int,
    provider: str | None = None,
    model: str | None = None,
    chunk_tokens: int | None = None,
) -> str:
    """Chronological left-fold summarization pipeline.

    1. Sort articles oldest-first
    2. Chunk into ~200K token batches
    3. Resume from last completed intermediate (crash resilience)
    4. Summarize each chunk
    5. Iteratively merge chunk summaries (left-fold, recency priority)
    6. Final synthesis pass for polish and references
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

    articles = db.get_articles(search_id, order_by="pub_date ASC")
    if not articles:
        raise ValueError(f"No articles found for search {search_id}")

    stats = db.get_search_stats(search_id)
    _log(f"Summarizing {len(articles)} articles "
         f"({stats['full_texts']} with full text), sorted oldest-first")
    _log(f"Provider: {provider}, Model: {model}")

    if chunk_tokens:
        config.MAX_CHUNK_TOKENS = chunk_tokens

    chunks = _chunk_articles(articles)
    n_chunks = len(chunks)
    _log(f"Split into {n_chunks} chunk(s) "
         f"(max {config.MAX_CHUNK_TOKENS} tokens/chunk)")

    # ── Phase 1: Summarize each chunk ────────────────────────────────
    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        existing = db.get_intermediate(search_id, "chunk", i)
        if existing and existing.get("summary_text"):
            _log(f"Chunk {i + 1}/{n_chunks}: found cached summary, skipping")
            chunk_summaries.append(existing["summary_text"])
            continue

        date_start, date_end = _date_range_for_chunk(chunk)
        _log(f"Chunk {i + 1}/{n_chunks}: {len(chunk)} articles "
             f"({date_start} to {date_end})")

        articles_text = format_articles_for_prompt(chunk)
        prompt = CHUNK_SUMMARY_PROMPT.format(articles_text=articles_text)
        summary = _call_llm(prompt, provider, model,
                            max_tokens=8192, temperature=0.3)

        db.insert_intermediate(
            search_id=search_id,
            step_type="chunk",
            step_index=i,
            summary_text=summary,
            model=model,
            date_range_start=date_start,
            date_range_end=date_end,
            articles_count=len(chunk),
        )
        chunk_summaries.append(summary)

        if i < n_chunks - 1:
            time.sleep(2)

    # ── Phase 2: Iterative merge (left-fold) ─────────────────────────
    if n_chunks == 1:
        accumulated = chunk_summaries[0]
        _log("Single chunk — skipping merge phase")
    else:
        accumulated = chunk_summaries[0]
        for i in range(1, n_chunks):
            merge_index = i - 1
            existing = db.get_intermediate(search_id, "merge", merge_index)
            if existing and existing.get("summary_text"):
                _log(f"Merge {merge_index + 1}/{n_chunks - 1}: "
                     f"found cached result, skipping")
                accumulated = existing["summary_text"]
                continue

            _log(f"Merge {merge_index + 1}/{n_chunks - 1}: "
                 f"merging accumulated + chunk {i + 1}")

            prompt = MERGE_PROMPT.format(
                accumulated_summary=accumulated,
                new_chunk_summary=chunk_summaries[i],
            )
            merged = _call_llm(prompt, provider, model,
                               max_tokens=12288, temperature=0.2)

            db.insert_intermediate(
                search_id=search_id,
                step_type="merge",
                step_index=merge_index,
                summary_text=merged,
                model=model,
            )
            accumulated = merged

            if i < n_chunks - 1:
                time.sleep(2)

    # ── Phase 3: Final synthesis ─────────────────────────────────────
    existing_final = db.get_intermediate(search_id, "final", 0)
    if existing_final and existing_final.get("summary_text"):
        _log("Final synthesis: found cached result, using it")
        return existing_final["summary_text"]

    _log("Running final synthesis pass...")
    prompt = SYNTHESIS_PROMPT.format(
        topic=topic,
        from_date=from_date,
        to_date=to_date,
        total_articles=len(articles),
        fulltext_count=stats["full_texts"],
        batch_summaries=accumulated,
    )
    summary_md = _call_llm(prompt, provider, model,
                           max_tokens=16384, temperature=0.2)

    db.insert_intermediate(
        search_id=search_id,
        step_type="final",
        step_index=0,
        summary_text=summary_md,
        model=model,
        articles_count=len(articles),
    )

    _log("Summarization complete")
    return summary_md
