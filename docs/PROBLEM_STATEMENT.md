# Problem Statement (as provided)

> Preserved verbatim from the assessment brief so the repository is self-describing
> and the implementation can be audited against the original requirements.

## BUILD

Build a replayable AI pipeline that ingests a small document corpus and a set of user questions, retrieves supporting evidence, generates grounded answers, scores answer quality, and produces a failure-analysis report.

This is not a chatbot demo. The evaluator will run your pipeline from a clean checkout, may replace the input files with equivalent fixtures using the same schema, and will verify that citation verification and core scoring logic are implemented deterministically in code rather than delegated entirely to the LLM.

Your solution should demonstrate production-quality engineering judgment for an AI engineer: clear staging, artifact persistence, schema validation, handling of imperfect retrieval/generation outputs, and explicit tradeoffs between recall, grounding, and answer quality.

The pipeline must preserve intermediate artifacts, enforce controlled vocabularies, log LLM calls, and separate deterministic evaluation from model-generated reasoning.

---

## INPUT FILES

The pipeline must read these files from disk:

- `documents.json`
- `questions.json`

The evaluator may replace these with equivalent data using the same schema. The implementation must not depend on exact IDs, document ordering, or wording from the sample fixture.

See `documents.json` and `questions.json` in the repository root for the sample fixtures.

---

## CONTROLLED VOCABULARIES

Define these vocabularies in code and validate outputs against them.

**Allowed answer labels:** `grounded`, `partially_grounded`, `not_grounded`, `insufficient_context`

**Allowed retrieval outcomes:** `exact_match`, `partial_match`, `miss`

**Allowed risk flags:** `policy_risk`, `compliance_risk`, `security_risk`, `hallucination_risk`, `none`

**Allowed failure modes:** `retrieval_failure`, `citation_failure`, `overclaim`, `unsupported_detail`, `policy_misread`, `no_failure`

---

## PIPELINE STAGES

```text
INIT
 -> DOCUMENTS_LOADED
 -> INDEX_BUILT
 -> RETRIEVAL_COMPLETE
 -> ANSWERS_GENERATED
 -> CITATIONS_VERIFIED
 -> ANSWERS_SCORED
 -> FAILURE_ANALYSIS_COMPLETE
 -> VALIDATION_COMPLETE
 -> RESULTS_FINALISED
```

Answer scoring and failure analysis must not run before deterministic citation verification has completed.

---

## MUST COMPLETE

1. **Corpus Preparation and Retrieval** — load `documents.json`, build a retrieval index, retrieve evidence for every question. `retrieval_outcome` assigned in deterministic code by comparing retrieved docs with `gold_doc_ids`. Save to `retrieval.json`.
2. **Grounded Answer Generation** — one LLM call per question receiving the question + retrieved evidence + instruction to answer only from context + cite supporting chunk IDs inline. May abstain explicitly. Save to `answers/{question_id}.json`.
3. **Deterministic Citation Verification** — implemented in code (do NOT ask the LLM whether an answer is grounded). Parse cited chunk IDs, verify they exist in retrieved evidence, check whether key factual claims are supported by cited text using deterministic heuristics, produce a groundedness label + explanation. Save to `verification.json`.
4. **Failure Analysis and Risk Flagging** — per-question analysis combining retrieval output and deterministic verification. `failure_mode` assigned in code using explicit rules; `risk_flag` from controlled vocabulary; distinguish retrieval problems from generation/citation problems. Save to `analysis.json`.
5. **Summary Report** — Markdown `report.md` with retrieval hit-rate vs `gold_doc_ids`, counts by groundedness label, counts by failure mode, highest-risk question(s), and 2–4 concrete production improvements. All numeric summaries from deterministic computation.

## SHOULD ATTEMPT

6. **Pairwise Retrieval Comparison** — compare two retrieval configurations (e.g. lexical vs embedding, chunked vs whole-document, top-k=2 vs top-k=4). Save to `retrieval_comparison.json` with config names, per-question retrieved docs, change in exact-match rate, qualitative tradeoff summary.
7. **Adversarial Question Handling** — 2–3 locally defined adversarial questions (ambiguous, multi-hop, or policy-sensitive). Run through the same pipeline. Save to `adversarial_questions.json` including whether the system abstained appropriately and why.

## STRETCH

8. **Minimal Service Interface** — e.g. `python app.py --question "..."` or a small HTTP endpoint that runs retrieval, answer generation, and verification for a single new question and returns structured JSON.

---

## REQUIRED ARTIFACTS

`documents.json`, `questions.json`, `retrieval.json`, `answers/`, `verification.json`, `analysis.json`, `report.md`, `llm_calls.jsonl`, plus `retrieval_comparison.json` and `adversarial_questions.json` if attempted.

### `llm_calls.jsonl`

One JSON object per LLM call with: `stage`, `question_id` (string|null), `timestamp` (ISO-8601), `provider`, `model`, `prompt_hash`, `input_artifacts` (paths), `output_artifact` (path). Separate records for each answer-generation call, any LLM-assisted report/comparison generation, and any adversarial-question answer calls. Deterministic steps that use no LLM must NOT be logged as LLM calls.

---

## VALIDATION REQUIREMENTS

A validation command (e.g. `make validate` / `python validate.py`) must check that: required artifacts exist; JSON files are valid; all questions processed; retrieval records exist for every question; answer files exist for every question; cited chunk IDs are well formed; verification labels use only the controlled vocabulary; failure modes and risk flags use only the controlled vocabularies; citation verification ran after answer generation; report metrics are consistent with machine-readable outputs; LLM call logs contain separate records for required stages.

---

## EXECUTION REQUIREMENTS

The evaluator will run the pipeline from a clean checkout. Generated artifacts may be deleted before evaluation. The evaluator may replace `documents.json` and `questions.json` with equivalent fixtures using the same schemas. Static precomputed outputs are not sufficient — the solution must actually rebuild the index, rerun retrieval, regenerate answers, rerun deterministic verification, and recreate the required artifacts.

---

## TECHNICAL CONSTRAINTS

- Retrieval must run from the supplied corpus at evaluation time.
- Core citation verification and answer labeling must be deterministic code.
- Controlled vocabularies must be defined in code and enforced.
- All questions must be processed.
- The system must tolerate missing or imperfect retrieval results without crashing.
- Do not fabricate policy or compliance rules not supported by the documents.
- Do not use private or proprietary data.
- Prefer simple, explicit, maintainable logic over opaque magic.
