"""Deterministic failure analysis and risk flagging.

Combines the deterministic retrieval outcome with the deterministic verification
result. Every label here comes from explicit rules in code -- never the LLM. Risk
flags are driven by the source_type of the cited documents (robust to fixture
swaps), with category_hint as a secondary signal.
"""
from collections import Counter

from .vocab import FailureMode, RiskFlag

# source_type -> domain risk mapping (data-driven, so swapped fixtures keep working).
COMPLIANCE_SOURCES = {"compliance"}
SECURITY_SOURCES = {"help_center"}            # account-security / identity docs
POLICY_SOURCES = {"policy", "operations", "runbook"}

_RECOMMENDED_FIX = {
    FailureMode.RETRIEVAL_FAILURE.value:
        "Gold document not retrieved. Improve retrieval: hybrid lexical+dense embeddings, "
        "better chunking, or higher top-k.",
    FailureMode.CITATION_FAILURE.value:
        "Citations missing or outside the retrieved set. Constrain generation to cite only "
        "provided chunk IDs and reject answers that fail the citation-in-set check.",
    FailureMode.OVERCLAIM.value:
        "Answer asserts claims unsupported by cited evidence. Tighten the grounding prompt and "
        "gate responses on claim-level verification before returning them.",
    FailureMode.UNSUPPORTED_DETAIL.value:
        "Some details are unsupported. Trim the answer to the evidence or retrieve additional "
        "supporting chunks.",
    FailureMode.POLICY_MISREAD.value:
        "Sensitive policy/compliance answer is not fully supported. Require exact-quote grounding "
        "and route this category to human review.",
    FailureMode.NO_FAILURE.value:
        "No action required; monitor.",
}


def _doc_of(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def _primary_source(answer, retrieval_record, doc_source):
    cited_docs = [_doc_of(c) for c in answer.get("cited_chunk_ids", [])]
    sources = [doc_source.get(d) for d in cited_docs if doc_source.get(d)]
    if not sources:
        sources = [doc_source.get(d) for d in retrieval_record["retrieved_doc_ids"] if doc_source.get(d)]
    if not sources:
        return None
    return Counter(sources).most_common(1)[0][0]


def analyse(answer, verification, retrieval_record, question, doc_source) -> dict:
    label = verification["answer_label"]
    outcome = retrieval_record["retrieval_outcome"]
    abstained = bool(answer.get("abstained", False))
    citation_present = verification["citation_ids_present"]
    all_in = verification["all_citations_in_retrieval_set"]
    unsupported_numeric = any(
        sc["method"] == "numeric_match" and not sc["supported"] for sc in verification["support_checks"]
    )

    source = _primary_source(answer, retrieval_record, doc_source)
    cat = question.category_hint
    is_compliance = source in COMPLIANCE_SOURCES or cat == "compliance"
    is_security = source in SECURITY_SOURCES or cat == "account_security"
    is_sensitive = is_compliance or is_security

    # --- failure_mode (explicit precedence) ---
    if abstained:
        # Abstaining is the safe behaviour and is never itself a failure.
        failure_mode = FailureMode.NO_FAILURE.value
    elif outcome == "miss":
        failure_mode = FailureMode.RETRIEVAL_FAILURE.value
    elif (not citation_present) or (not all_in):
        failure_mode = FailureMode.CITATION_FAILURE.value
    elif label in ("not_grounded", "partially_grounded"):
        if is_sensitive:
            failure_mode = FailureMode.POLICY_MISREAD.value
        elif label == "not_grounded":
            failure_mode = FailureMode.OVERCLAIM.value
        else:
            failure_mode = FailureMode.UNSUPPORTED_DETAIL.value
    else:
        failure_mode = FailureMode.NO_FAILURE.value

    # --- risk_flag ---
    if abstained or (label == "grounded" and failure_mode == FailureMode.NO_FAILURE.value):
        risk = RiskFlag.NONE.value
    elif is_compliance:
        risk = RiskFlag.COMPLIANCE_RISK.value
    elif is_security:
        risk = RiskFlag.SECURITY_RISK.value
    elif unsupported_numeric or label == "not_grounded":
        risk = RiskFlag.HALLUCINATION_RISK.value
    elif source in POLICY_SOURCES:
        risk = RiskFlag.POLICY_RISK.value
    else:
        risk = RiskFlag.HALLUCINATION_RISK.value

    fix = _RECOMMENDED_FIX[failure_mode]
    if abstained and outcome == "exact_match":
        fix = ("Model abstained although gold evidence was retrieved; consider calibrating the "
               "abstention threshold so available evidence is used.")

    return {
        "question_id": question.question_id,
        "retrieval_outcome": outcome,
        "answer_label": label,
        "failure_mode": failure_mode,
        "risk_flag": risk,
        "recommended_fix": fix,
    }
