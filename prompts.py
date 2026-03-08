"""Structured prompt templates for medical literature review summarization."""


CHUNK_SUMMARY_PROMPT = """You are a medical research analyst. Analyze the following batch of PubMed articles and produce a structured summary.

For each article, you have the PMID, title, authors, journal, publication date, abstract, and possibly full text.

INSTRUCTIONS:
- Cite every factual claim using the format [PMID: XXXXXXXX]
- Preserve the PMID identifiers exactly as given
- Focus on extracting key findings, methodology, and conclusions
- Note the study type (RCT, cohort, case series, review, meta-analysis, etc.)
- Flag any limitations mentioned by the authors

Produce your summary in these sections (skip any section with no relevant data):

1. PATHOBIOLOGY: Disease mechanisms, pathophysiology, etiology, natural history
2. MOLECULAR AND IMMUNOLOGIC CHARACTERISTICS: Biomarkers, genomic alterations, immune profiling, tumor microenvironment, molecular subtypes
3. TREATMENT BY STAGE: Standard-of-care therapies organized by disease stage (early, locally advanced, metastatic)
4. NEOADJUVANT AND ADJUVANT APPROACHES: Pre-surgical and post-surgical treatment strategies, perioperative regimens
5. EXPERIMENTAL APPROACHES: Clinical trials, novel agents, emerging therapies, combination strategies, immunotherapy, targeted therapy
6. LIMITATIONS: Sample size concerns, reproducibility issues, unsupported hypotheses, selection bias, short follow-up, methodological weaknesses
7. KEY FINDINGS: The most important results from this batch, with citations

---
ARTICLES:
{articles_text}
---

Respond in well-structured markdown with the section headers above."""


SYNTHESIS_PROMPT = """You are a senior medical research analyst writing a comprehensive literature review. You have been given summaries from multiple batches of PubMed articles on the topic: "{topic}"

Date range: {from_date} to {to_date}
Total articles analyzed: {total_articles}
Articles with full text available: {fulltext_count}

Below are the batch summaries. Synthesize them into a single, cohesive, publication-quality literature review.

REQUIREMENTS:
- Maintain ALL citations in [PMID: XXXXXXXX] format
- Do not invent or fabricate any citations
- Organize the review into the following sections:

## 1. Overview
Scope of the review, number of articles, date range, search methodology.

## 2. Pathobiology
Disease mechanisms, pathophysiology, etiology, natural history. Synthesize findings across studies.

## 3. Molecular and Immunologic Characteristics
Biomarkers, genomic alterations (mutations, amplifications, fusions), immune profiling, tumor microenvironment, molecular subtypes and their clinical relevance.

## 4. Treatment by Stage
Standard-of-care therapies organized by disease stage. Include response rates, survival data, and guideline recommendations where available.

## 5. Neoadjuvant and Adjuvant Approaches
Pre-surgical and post-surgical treatment strategies. Include pathologic response rates and survival outcomes.

## 6. Experimental Approaches
Clinical trials (with phase and NCT numbers if mentioned), novel agents, emerging therapies, combination strategies, immunotherapy, targeted therapy, and other innovative approaches.

## 7. Limitations of Current Evidence
Sample size concerns, reproducibility issues, unsupported hypotheses, selection bias, short follow-up periods, heterogeneous study designs, publication bias. Be specific about which studies have which limitations.

## 8. Most Impactful Studies
Rank the top studies by citation count and scientific impact. For each, provide:
- Full citation (authors, title, journal, year)
- PMID
- Citation count (if available)
- Key finding and why it is impactful

## 9. Most Recent Findings
Summarize the most recent publications (last 6-12 months of the search range). Highlight any paradigm shifts or emerging trends.

## 10. References
Complete numbered reference list in Vancouver/NLM citation style:
Author(s). Title. Journal. Year;Volume(Issue):Pages. PMID: XXXXXXXX.

---
BATCH SUMMARIES:
{batch_summaries}
---

Write a thorough, well-organized literature review in markdown format. Be comprehensive but concise. Every claim must be cited."""


def format_articles_for_prompt(articles: list[dict]) -> str:
    """Format a list of articles into text for the LLM prompt."""
    parts = []
    for a in articles:
        lines = [
            f"### PMID: {a.get('pmid', 'N/A')}",
            f"**Title:** {a.get('title', 'N/A')}",
            f"**Authors:** {a.get('authors', 'N/A')}",
            f"**Journal:** {a.get('journal', 'N/A')}",
            f"**Date:** {a.get('pub_date', 'N/A')}",
            f"**DOI:** {a.get('doi', 'N/A')}",
            f"**Citation Count:** {a.get('citation_count', 0)}",
        ]

        mesh = a.get("mesh_terms", "")
        if mesh:
            if isinstance(mesh, str):
                lines.append(f"**MeSH Terms:** {mesh}")
            else:
                lines.append(f"**MeSH Terms:** {', '.join(mesh)}")

        abstract = a.get("abstract", "")
        if abstract:
            lines.append(f"\n**Abstract:** {abstract}")

        fulltext = a.get("text_content", "")
        if fulltext:
            truncated = fulltext[:8000]
            if len(fulltext) > 8000:
                truncated += "\n[... full text truncated ...]"
            lines.append(f"\n**Full Text:**\n{truncated}")

        parts.append("\n".join(lines))

    return "\n\n---\n\n".join(parts)
