# Replayable RAG Evaluation Pipeline

A small, **replayable** AI pipeline that ingests a document corpus and a set of user
questions, retrieves supporting evidence, generates **grounded** answers with one LLM
call each, then **deterministically** verifies citations, scores answer quality, and
produces a failure-analysis report.

The design goal is production engineering judgment, not a chatbot demo: clear staging,
artifact persistence, schema/vocabulary validation, graceful handling of imperfect
retrieval/generation, and a hard separation between **model-generated reasoning**
(answer text) and **deterministic evaluation** (everything that scores or labels).

> **Zero required dependencies.** The entire pipeline — retrieval, verification,
> scoring, analysis, report, validation, and tests — runs on the Python standard
> library. The real LLM is called over stdlib `urllib`. See
> [`docs/DECISIONS.md`](docs/DECISIONS.md) for why.

## Quickstart

```bash
# 1. Deterministic run — no API key, no network, no installs:
python3 run.py --mock-llm
python3 validate.py            # -> VALIDATION PASSED

# 2. Real LLM run (OpenRouter). Put your key in .env (gitignored):
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
python3 run.py
python3 validate.py

# 3. Tests (stdlib unittest, no installs):
python3 -m unittest discover -s tests -v

# 4. Single-question service (STRETCH):
python3 app.py --question "I got HTTP 429 errors. What are the limits and what should I do?"
```

`make run`, `make run-mock`, `make validate`, `make test`, `make clean` wrap these.

## How it works

The pipeline is a state machine that enforces this exact order in code and persists a
timestamped transition log to `run_manifest.json`:

```
INIT → DOCUMENTS_LOADED → INDEX_BUILT → RETRIEVAL_COMPLETE → ANSWERS_GENERATED
     → CITATIONS_VERIFIED → ANSWERS_SCORED → FAILURE_ANALYSIS_COMPLETE
     → VALIDATION_COMPLETE → RESULTS_FINALISED
```

Scoring and failure analysis **cannot** run before deterministic citation verification —
`validate.py` reads the manifest and fails if `CITATIONS_VERIFIED` did not follow
`ANSWERS_GENERATED`.

| Stage | What happens | Deterministic? |
|---|---|---|
| Retrieval | Pure-Python **BM25** over sentence chunks (`{doc_id}::{i}`). `retrieval_outcome` (`exact_match`/`partial_match`/`miss`) is computed by comparing retrieved docs to `gold_doc_ids`. | ✅ |
| Answer generation | **One LLM call per question**, instructed to answer only from the evidence and cite chunk IDs inline. May abstain explicitly. The **only** LLM-touching stage. | ❌ (LLM) |
| Citation verification | Parse cited IDs → confirm all are in the retrieved set → require every numeric/code fact to appear in cited text → content-token overlap ratio → groundedness label. **Never asks the LLM whether the answer is grounded.** | ✅ |
| Failure analysis | `failure_mode` + `risk_flag` from explicit rules, driven by the cited docs' `source_type` (robust to fixture swaps). | ✅ |
| Report | `metrics.json` + `report.md`, all numbers computed in code. | ✅ |

## Controlled vocabularies (enforced in code)

Defined once in [`pipeline/vocab.py`](pipeline/vocab.py) and validated everywhere:

- **answer labels:** `grounded`, `partially_grounded`, `not_grounded`, `insufficient_context`
- **retrieval outcomes:** `exact_match`, `partial_match`, `miss`
- **risk flags:** `policy_risk`, `compliance_risk`, `security_risk`, `hallucination_risk`, `none`
- **failure modes:** `retrieval_failure`, `citation_failure`, `overclaim`, `unsupported_detail`, `policy_misread`, `no_failure`

## Deterministic verification heuristic

The verifier ([`pipeline/verification.py`](pipeline/verification.py)) is intentionally a
simple, documented heuristic — not full NL entailment:

1. **Citation integrity:** citations present (when not abstaining) and every cited chunk
   ID belongs to that question's retrieved evidence set.
2. **Numeric/code grounding:** every number in the answer (e.g. `120`, `429`, `10,000`,
   `24`) must appear, comma-normalised, in the cited chunk text.
3. **Lexical grounding:** content-token overlap ratio between answer and cited text, with
   light stemming so a paraphrase (`products` ↔ `product`) still matches.

Label rules (explicit): abstain → `insufficient_context`; missing/out-of-set citations →
`not_grounded`; all numerics supported **and** overlap ≥ threshold → `grounded`; partial
support → `partially_grounded`; otherwise `not_grounded`.

## LLM providers

Provider is auto-detected at call time and the chosen provider is logged honestly:

1. `OPENROUTER_API_KEY` → **OpenRouter** (OpenAI-compatible, via stdlib `urllib`; default model `openai/gpt-4o-mini`, override with `LLM_MODEL`).
2. `ANTHROPIC_API_KEY` → **Anthropic** SDK (lazy import).
3. none / `--mock-llm` / any API error → **deterministic mock** (synthesises a
   citation-bearing answer from the top chunks; abstains when no evidence). This is why
   the pipeline always runs from a clean checkout.

The API key is read only from the environment (`.env` is gitignored) and is never logged
or persisted — `llm_calls.jsonl` records only `provider`, `model`, and a `prompt_hash`.

## Artifacts produced

`retrieval.json`, `answers/{question_id}.json`, `verification.json`, `analysis.json`,
`report.md`, `metrics.json`, `run_manifest.json`, `llm_calls.jsonl`,
`retrieval_comparison.json` (SHOULD #6: sentence-chunk vs whole-document), and
`adversarial_questions.json` (SHOULD #7: 3 locally-defined adversarial questions with an
abstention-appropriateness judgement).

## Validation

`python3 validate.py` checks: required artifacts exist; JSON is valid; all questions
processed; a retrieval and answer record exists per question; cited chunk IDs are
well-formed; verification labels, failure modes and risk flags use only the controlled
vocabularies; **citation verification ran after answer generation**; **report metrics are
recomputed and cross-checked** against the machine-readable artifacts; and `llm_calls.jsonl`
contains separate records for each answer-generation call (and each adversarial call).

## Repository layout

```
documents.json, questions.json    input corpus + questions (sample fixtures; swappable)
adversarial.json                  locally-defined adversarial questions
run.py                            pipeline entry point
validate.py                       standalone validation command
app.py                            single-question service interface (STRETCH)
pipeline/                         vocab, config, corpus, retrieval, llm, answering,
                                  verification, analysis, report, state, runner
tests/                            stdlib unittest suite (26 tests)
docs/                             PROBLEM_STATEMENT.md, DECISIONS.md (ADR log)
```

## What I'd do next for production

See the generated [`report.md`](report.md) for the full list. Headlines: hybrid
(BM25 + dense embedding) retrieval with a re-ranker; replace the lexical support heuristic
with claim-level NLI entailment; add answer-relevance scoring and abstention calibration;
and gate compliance/security categories behind exact-quote grounding + human review.
