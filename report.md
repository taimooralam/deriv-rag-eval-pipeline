# RAG Evaluation Report

_All numeric summaries below are computed deterministically from the machine-readable artifacts (`retrieval.json`, `verification.json`, `analysis.json`). See `metrics.json` for the structured form; `validate.py` recomputes and cross-checks these numbers._

## Retrieval

- Questions processed: **5**
- Exact-match hit-rate vs `gold_doc_ids`: **100%**
- Any-overlap recall (exact + partial): **100%**

### Retrieval outcomes

- `exact_match`: 5

## Answer quality

### Groundedness labels

- `grounded`: 4
- `partially_grounded`: 1

## Failure analysis

### Failure modes

- `no_failure`: 4
- `unsupported_detail`: 1

### Risk flags

- `none`: 4
- `hallucination_risk`: 1

## Highest-risk question(s)

- **Q1** — risk `hallucination_risk`, failure `unsupported_detail`

## Top improvements for production

1. Add hybrid retrieval (BM25 + dense embeddings) with a cross-encoder re-ranker. Lexical retrieval is sufficient for this tiny corpus, but semantic recall degrades as the corpus grows and paraphrases diverge from document wording.
2. Replace the substring/numeric support heuristic with a claim-level NLI entailment model (LLM or a small fine-tuned classifier) for the verifier, keeping the deterministic checks as a fast guardrail/pre-filter.
3. Calibrate abstention and add answer-relevance scoring: today the verifier checks grounding to cited text but not whether the answer addresses the question, so an on-topic-but-wrong or off-topic-but-grounded answer can slip through.
4. Gate compliance/security categories behind mandatory human review and exact-quote grounding, and expand the corpus + add regression fixtures so retrieval/verification thresholds are tuned against labelled data rather than hand-picked constants.
