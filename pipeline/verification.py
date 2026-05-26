"""Deterministic citation verification (NO LLM involvement).

For each answer we:
  1. check that cited chunk IDs are present and all belong to the retrieved evidence set;
  2. extract numeric/code facts from the answer and require each to appear (comma-normalised)
     in the cited chunk text;
  3. compute a content-token overlap ratio between the answer and the cited text;
  4. assign a groundedness label from explicit rules.
This is an intentionally simple, documented heuristic -- not full NL entailment.
"""
import re

from .vocab import AnswerLabel

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+")
# Inline citation markers like [D1::0] are metadata, not claims — strip before analysis
# so chunk-id digits don't pollute numeric/lexical grounding checks.
_CITE_STRIP = re.compile(r"\[[^\]]*\]")

# Small stopword list so the overlap ratio reflects substantive content tokens.
_STOPWORDS = set(
    "a an the and or of to in on for with is are be can may should must not no "
    "you your user users this that it if their they them as at by from we our will "
    "would do does done been being have has had what when where who why how about "
    "into over per any all only based use using".split()
)


def _norm_num(s: str) -> str:
    return s.replace(",", "")


def tokenize(text: str):
    return _WORD.findall((text or "").lower())


def stem(token: str) -> str:
    """Lemmatisation-lite: collapse common plural/inflection endings so a paraphrased
    answer ('products') still matches the evidence ('product'). Cheap and deterministic;
    full lemmatisation/NLI is the documented production upgrade."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def content_tokens(text: str):
    return [stem(t) for t in tokenize(text) if t not in _STOPWORDS and len(t) >= 3]


def stem_set(text: str):
    return {stem(t) for t in tokenize(text)}


def numbers_in(text: str):
    return {_norm_num(n) for n in _NUM.findall(text or "")}


def extract_numeric_claims(answer_text: str):
    """Return list of (normalised_number, human_readable_claim)."""
    claims = []
    for m in _NUM.finditer(answer_text or ""):
        num = m.group(0)
        tail = (answer_text or "")[m.end(): m.end() + 30]
        ctx = re.findall(r"[A-Za-z%]+", tail)[:3]
        claim = (num + " " + " ".join(ctx)).strip()
        claims.append((_norm_num(num), claim))
    return claims


def verify_answer(answer: dict, retrieval_record: dict, cfg) -> dict:
    cited = answer.get("cited_chunk_ids", [])
    abstained = bool(answer.get("abstained", False))
    text_by_id = {c["chunk_id"]: c["text"] for c in retrieval_record["top_k_chunks"]}
    retrieval_ids = set(text_by_id)

    citation_ids_present = bool(cited)
    all_in = all(c in retrieval_ids for c in cited)  # True (vacuously) when no citations

    cited_text = " ".join(text_by_id.get(c, "") for c in cited)
    cited_numbers = numbers_in(cited_text)
    cited_tokens = stem_set(cited_text)

    # Analyse the prose only — drop inline [chunk::id] markers so their digits/letters
    # are not mistaken for factual claims.
    answer_prose = _CITE_STRIP.sub(" ", answer.get("answer", ""))

    support_checks = []

    # (1) numeric / code facts
    numeric_claims = extract_numeric_claims(answer_prose)
    supported_numeric = 0
    for norm, claim in numeric_claims:
        supported = norm in cited_numbers
        supporting_ids = [c for c in cited if norm in numbers_in(text_by_id.get(c, ""))]
        support_checks.append(
            {
                "claim": claim,
                "supported": supported,
                "supporting_chunk_ids": supporting_ids,
                "method": "numeric_match",
            }
        )
        if supported:
            supported_numeric += 1
    numeric_ratio = (supported_numeric / len(numeric_claims)) if numeric_claims else 1.0

    # (2) overall lexical grounding
    ans_tokens = content_tokens(answer_prose)
    overlap = sum(1 for t in ans_tokens if t in cited_tokens)
    lexical_ratio = (overlap / len(ans_tokens)) if ans_tokens else (1.0 if abstained else 0.0)
    support_checks.append(
        {
            "claim": "overall lexical grounding of answer against cited evidence",
            "supported": lexical_ratio >= cfg.lexical_grounded_threshold,
            "supporting_chunk_ids": list(cited),
            "method": "token_overlap_ratio",
            "ratio": round(lexical_ratio, 3),
        }
    )

    # (3) label assignment (explicit, deterministic)
    if abstained:
        label = AnswerLabel.INSUFFICIENT_CONTEXT.value
        explanation = "Answer abstained explicitly; labelled insufficient_context."
    elif not citation_ids_present:
        label = AnswerLabel.NOT_GROUNDED.value
        explanation = "Non-abstaining answer provided no citations."
    elif not all_in:
        label = AnswerLabel.NOT_GROUNDED.value
        explanation = "One or more cited chunk IDs are absent from the retrieved evidence set."
    elif numeric_ratio >= 1.0 and lexical_ratio >= cfg.lexical_grounded_threshold:
        label = AnswerLabel.GROUNDED.value
        explanation = (
            f"All {len(numeric_claims)} numeric claim(s) supported by cited text; "
            f"lexical grounding {lexical_ratio:.2f} >= {cfg.lexical_grounded_threshold}."
        )
    elif numeric_ratio >= 0.5 or lexical_ratio >= cfg.lexical_partial_threshold:
        label = AnswerLabel.PARTIALLY_GROUNDED.value
        explanation = (
            f"Partial support: numeric_ratio={numeric_ratio:.2f}, lexical_ratio={lexical_ratio:.2f}."
        )
    else:
        label = AnswerLabel.NOT_GROUNDED.value
        explanation = (
            f"Insufficient support: numeric_ratio={numeric_ratio:.2f}, lexical_ratio={lexical_ratio:.2f}."
        )

    return {
        "question_id": answer["question_id"],
        "citation_ids_present": citation_ids_present,
        "all_citations_in_retrieval_set": bool(all_in),
        "support_checks": support_checks,
        "answer_label": label,
        "explanation": explanation,
    }
