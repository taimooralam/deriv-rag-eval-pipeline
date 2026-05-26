# Design Decisions (lightweight ADR log)

Recorded as decisions were made. Each entry: context → decision → rationale → consequences.

## ADR-000: Time-budget tradeoff (governs everything below)
- **Context:** Hard 60-minute timed assessment; scoring weighs both quality and completion time.
- **Decision:** Optimize *quality-per-minute*. Strict priority order: MUST (corpus/retrieval, answer gen, deterministic verification, failure analysis, report) + `validate.py` + `llm_calls.jsonl` first; then SHOULD (retrieval comparison, adversarial); then STRETCH (service interface). Prefer explicit, legible code over breadth.
- **Consequence:** Some production niceties (NLI entailment, embeddings, async, packaging) are deliberately deferred and called out as "next steps" rather than half-built.

## ADR-001: Language — Python 3.11+
Examples in the brief (`python validate.py`, `python app.py`) and the wider project ecosystem are Python. No reason to diverge.

## ADR-002: Minimal dependency surface — stdlib-only runtime; no LLM framework
- **Decision:** The entire deterministic pipeline + the mock path run on the **Python standard library only**. `anthropic` is imported lazily and only when a real LLM call is made. No LangChain/LlamaIndex; no numpy/sklearn; no pydantic.
- **Rationale:** The brief grades *legible, deterministic* logic ("prefer simple, explicit, maintainable logic over opaque magic") and *clean-checkout reliability*. A framework hides exactly the retrieval/verification/scoring logic being graded, and every added dependency is an install failure mode on the evaluator's machine. Zero runtime deps means the pipeline runs offline, keyless, with no `pip install`. Production quality is shown through module boundaries, an enforced state machine, schema/vocabulary validation, structured LLM logging, graceful degradation, and tests — not dependency count.
- **Considered & rejected:** LangChain (experience with it exists, but it works against demonstrability here); pydantic (nice, but adds an install dependency for validation we can express explicitly in stdlib).

## ADR-003: Retrieval — pure-Python BM25 over sentence chunks
- **Decision:** Primary retriever is a hand-written BM25 (Okapi) over sentence-level chunks (`chunk_id = "{doc_id}::{i}"`). Deterministic tie-breaking by `(-score, chunk_id)`.
- **Rationale:** Tiny corpus → lexical retrieval fully solves recall here; sentence granularity makes citation support-checking meaningful. Zero deps, fully unit-testable, identical results across machines (no float/model drift).
- **Embeddings / vector search:** The correct production foundation at scale is **hybrid (BM25 + dense embeddings + reranking)**. It is intentionally *not* built here — at 5 short docs it adds risk (heavy install or network, float nondeterminism) with no recall benefit. It is documented as the #1 production next-step and represented in the retrieval comparison conceptually.

## ADR-004: Retrieval-outcome rule (deterministic)
`exact_match` = all `gold_doc_ids` ⊆ retrieved; `partial_match` = nonempty-but-incomplete overlap; `miss` = no overlap (or no gold defined). Computed in code, never by the LLM.

## ADR-005: LLM usage — one grounded call per question, temp 0, deterministic fallback
- One LLM call per answer; `temperature=0`; structured JSON output. If `ANTHROPIC_API_KEY` is absent, `anthropic` is not installed, or `--mock-llm` is set, a **deterministic mock** synthesises a citation-bearing answer from the top retrieved chunks (abstaining when no evidence). Every call (real or mock) is logged to `llm_calls.jsonl` with the provider tagged honestly. The LLM is used **only** for answer generation — never for grounding judgments.

## ADR-006: Deterministic citation verification heuristic
- Parse cited chunk IDs → confirm presence and that all are in the retrieved set → extract **numeric/code facts** (e.g. `120`, `429`, `10,000`, `24`) and require each to appear (comma-normalized) in cited text → compute a **content-token overlap ratio** between answer and cited text. Label rules (explicit, in code): abstain → `insufficient_context`; missing/out-of-set citations → `not_grounded`; all numerics supported + overlap ≥ threshold → `grounded`; partial support → `partially_grounded`; else → `not_grounded`. Documented heuristic; not full NLI by design.

## ADR-007: Failure-mode & risk-flag rules driven by data, not the prompt
- `failure_mode` precedence: `retrieval_failure` (gold missed) → `citation_failure` (missing/out-of-set citations) → on sensitive source `policy_misread`, else `overclaim`/`unsupported_detail` → `no_failure`. `risk_flag` derived from the **`source_type` of cited docs** (compliance→compliance_risk, help_center→security_risk, policy/operations/runbook→policy_risk) plus `hallucination_risk` for unsupported content — robust to fixture swaps (doesn't trust `category_hint` alone). Correct abstention is `no_failure`/`none`.

## ADR-008: Stage enforcement & report consistency
- A `StateMachine` enforces the exact stage order in code and writes timestamped transitions to `run_manifest.json`; this proves `CITATIONS_VERIFIED` ran after `ANSWERS_GENERATED`. `report.md` is generated from a computed `metrics.json`; `validate.py` independently recomputes metrics from primary artifacts and checks consistency.
