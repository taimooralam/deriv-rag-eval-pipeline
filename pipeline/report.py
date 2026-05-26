"""Deterministic metrics computation + Markdown report generation.

All numbers come from compute_metrics() over the machine-readable artifacts. report.md
is rendered from the same metrics dict that is persisted to metrics.json, so the two
can never disagree (validate.py recomputes independently to confirm).
"""
from collections import Counter

# Higher number => higher severity, used to pick the highest-risk question(s).
_RISK_SEVERITY = {
    "compliance_risk": 4,
    "security_risk": 3,
    "hallucination_risk": 2,
    "policy_risk": 1,
    "none": 0,
}

IMPROVEMENTS = [
    "Add hybrid retrieval (BM25 + dense embeddings) with a cross-encoder re-ranker. Lexical "
    "retrieval is sufficient for this tiny corpus, but semantic recall degrades as the corpus "
    "grows and paraphrases diverge from document wording.",
    "Replace the substring/numeric support heuristic with a claim-level NLI entailment model "
    "(LLM or a small fine-tuned classifier) for the verifier, keeping the deterministic checks "
    "as a fast guardrail/pre-filter.",
    "Calibrate abstention and add answer-relevance scoring: today the verifier checks grounding "
    "to cited text but not whether the answer addresses the question, so an on-topic-but-wrong or "
    "off-topic-but-grounded answer can slip through.",
    "Gate compliance/security categories behind mandatory human review and exact-quote grounding, "
    "and expand the corpus + add regression fixtures so retrieval/verification thresholds are "
    "tuned against labelled data rather than hand-picked constants.",
]


def compute_metrics(retrieval_records, verification_records, analysis_records, n_questions) -> dict:
    outcomes = Counter(r["retrieval_outcome"] for r in retrieval_records)
    labels = Counter(v["answer_label"] for v in verification_records)
    failure_modes = Counter(a["failure_mode"] for a in analysis_records)
    risks = Counter(a["risk_flag"] for a in analysis_records)

    exact = outcomes.get("exact_match", 0)
    partial = outcomes.get("partial_match", 0)
    n = n_questions or 1

    ranked = sorted(analysis_records, key=lambda a: -_RISK_SEVERITY.get(a["risk_flag"], 0))
    highest_risk = [
        {"question_id": a["question_id"], "risk_flag": a["risk_flag"], "failure_mode": a["failure_mode"]}
        for a in ranked
        if a["risk_flag"] != "none"
    ]

    return {
        "n_questions": n_questions,
        "retrieval_hit_rate_exact": round(exact / n, 4),
        "retrieval_recall_any": round((exact + partial) / n, 4),
        "retrieval_outcomes": dict(outcomes),
        "answer_label_counts": dict(labels),
        "failure_mode_counts": dict(failure_modes),
        "risk_flag_counts": dict(risks),
        "highest_risk_questions": highest_risk,
    }


def _counts_block(title, counts):
    lines = [f"### {title}", ""]
    if not counts:
        lines.append("_none_")
    else:
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{k}`: {v}")
    lines.append("")
    return lines


def render_report(metrics: dict) -> str:
    m = metrics
    lines = [
        "# RAG Evaluation Report",
        "",
        "_All numeric summaries below are computed deterministically from the machine-readable "
        "artifacts (`retrieval.json`, `verification.json`, `analysis.json`). See `metrics.json` "
        "for the structured form; `validate.py` recomputes and cross-checks these numbers._",
        "",
        "## Retrieval",
        "",
        f"- Questions processed: **{m['n_questions']}**",
        f"- Exact-match hit-rate vs `gold_doc_ids`: **{m['retrieval_hit_rate_exact']:.0%}**",
        f"- Any-overlap recall (exact + partial): **{m['retrieval_recall_any']:.0%}**",
        "",
    ]
    lines += _counts_block("Retrieval outcomes", m["retrieval_outcomes"])
    lines += ["## Answer quality", ""]
    lines += _counts_block("Groundedness labels", m["answer_label_counts"])
    lines += ["## Failure analysis", ""]
    lines += _counts_block("Failure modes", m["failure_mode_counts"])
    lines += _counts_block("Risk flags", m["risk_flag_counts"])

    lines += ["## Highest-risk question(s)", ""]
    if not m["highest_risk_questions"]:
        lines.append("_No questions flagged with risk; all answers grounded with no failures._")
    else:
        for h in m["highest_risk_questions"]:
            lines.append(
                f"- **{h['question_id']}** — risk `{h['risk_flag']}`, failure `{h['failure_mode']}`"
            )
    lines.append("")

    lines += ["## Top improvements for production", ""]
    for i, imp in enumerate(IMPROVEMENTS, 1):
        lines.append(f"{i}. {imp}")
    lines.append("")
    return "\n".join(lines)
