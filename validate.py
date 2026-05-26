#!/usr/bin/env python3
"""Validation command.

Checks that the pipeline produced consistent, schema-conformant artifacts. Used both
standalone (`python3 validate.py`) and in-process by the runner. Every check below maps
to a requirement in the brief's VALIDATION REQUIREMENTS section.
"""
import json
import os
import re
import sys

from pipeline.config import Config
from pipeline.report import compute_metrics
from pipeline.vocab import (
    ANSWER_LABELS,
    FAILURE_MODES,
    RETRIEVAL_OUTCOMES,
    RISK_FLAGS,
)

_CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+::[A-Za-z0-9_.\-]+$")
_LLM_RECORD_KEYS = {
    "stage", "question_id", "timestamp", "provider", "model",
    "prompt_hash", "input_artifacts", "output_artifact",
}


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_all(cfg):
    """Return (ok: bool, errors: list[str])."""
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    # 1. Required artifacts exist
    required = [
        cfg.documents_path, cfg.questions_path, cfg.retrieval_path, cfg.verification_path,
        cfg.analysis_path, cfg.report_path, cfg.metrics_path, cfg.llm_log_path, cfg.manifest_path,
    ]
    for path in required:
        check(os.path.exists(path), f"missing required artifact: {path}")
    check(os.path.isdir(cfg.answers_dir), f"missing answers directory: {cfg.answers_dir}")
    if errors:
        return False, errors  # cannot continue without the core artifacts

    # 2. JSON validity
    try:
        documents = _load(cfg.documents_path)
        questions = _load(cfg.questions_path)
        retrieval = _load(cfg.retrieval_path)
        verification = _load(cfg.verification_path)
        analysis = _load(cfg.analysis_path)
        metrics = _load(cfg.metrics_path)
    except Exception as exc:
        return False, [f"invalid JSON in a core artifact: {exc}"]

    question_ids = [q["question_id"] for q in questions]
    qid_set = set(question_ids)

    # 3/4. All questions processed; retrieval record per question
    retrieval_ids = {r["question_id"] for r in retrieval}
    check(retrieval_ids == qid_set, f"retrieval.json questions {retrieval_ids} != questions {qid_set}")
    for r in retrieval:
        check("retrieval_outcome" in r and r["retrieval_outcome"] in RETRIEVAL_OUTCOMES,
              f"bad retrieval_outcome for {r.get('question_id')}: {r.get('retrieval_outcome')}")

    # 5. Answer file per question + well-formed cited chunk IDs
    for qid in question_ids:
        ans_path = os.path.join(cfg.answers_dir, f"{qid}.json")
        if not os.path.exists(ans_path):
            errors.append(f"missing answer file: {ans_path}")
            continue
        try:
            ans = _load(ans_path)
        except Exception as exc:
            errors.append(f"invalid answer JSON {ans_path}: {exc}")
            continue
        for cid in ans.get("cited_chunk_ids", []):
            check(bool(_CHUNK_ID_RE.match(str(cid))), f"malformed cited chunk id in {qid}: {cid!r}")
        if not ans.get("abstained", False):
            check(bool(ans.get("cited_chunk_ids")),
                  f"non-abstaining answer {qid} has no cited chunk ids")

    # 6. Verification labels in controlled vocabulary; one record per question
    verif_ids = {v["question_id"] for v in verification}
    check(verif_ids == qid_set, f"verification.json questions {verif_ids} != questions {qid_set}")
    for v in verification:
        check(v.get("answer_label") in ANSWER_LABELS,
              f"verification label out of vocab for {v.get('question_id')}: {v.get('answer_label')}")

    # 7. Failure modes & risk flags in controlled vocabularies; one record per question
    analysis_ids = {a["question_id"] for a in analysis}
    check(analysis_ids == qid_set, f"analysis.json questions {analysis_ids} != questions {qid_set}")
    for a in analysis:
        check(a.get("failure_mode") in FAILURE_MODES,
              f"failure_mode out of vocab for {a.get('question_id')}: {a.get('failure_mode')}")
        check(a.get("risk_flag") in RISK_FLAGS,
              f"risk_flag out of vocab for {a.get('question_id')}: {a.get('risk_flag')}")

    # 8. Citation verification ran after answer generation (from the manifest)
    try:
        manifest = _load(cfg.manifest_path)
        order = [t["stage"] for t in manifest["transitions"]]
        check("ANSWERS_GENERATED" in order and "CITATIONS_VERIFIED" in order,
              "manifest missing ANSWERS_GENERATED/CITATIONS_VERIFIED stages")
        if "ANSWERS_GENERATED" in order and "CITATIONS_VERIFIED" in order:
            check(order.index("CITATIONS_VERIFIED") > order.index("ANSWERS_GENERATED"),
                  "CITATIONS_VERIFIED did not run after ANSWERS_GENERATED")
    except Exception as exc:
        errors.append(f"could not validate stage ordering: {exc}")

    # 9. Report metrics consistent with machine-readable outputs
    recomputed = compute_metrics(retrieval, verification, analysis, len(questions))
    for key in ("n_questions", "retrieval_hit_rate_exact", "retrieval_outcomes",
                "answer_label_counts", "failure_mode_counts", "risk_flag_counts"):
        check(metrics.get(key) == recomputed.get(key),
              f"metrics.json[{key}]={metrics.get(key)} != recomputed {recomputed.get(key)}")

    # 10. LLM call log: valid records + separate records for required stages
    answer_gen_qids = set()
    adversarial_qids = set()
    try:
        with open(cfg.llm_log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                missing = _LLM_RECORD_KEYS - set(rec)
                check(not missing, f"llm_calls.jsonl line {i} missing keys: {missing}")
                if rec.get("stage") == "ANSWERS_GENERATED":
                    answer_gen_qids.add(rec.get("question_id"))
                elif rec.get("stage") == "ADVERSARIAL_ANSWERS_GENERATED":
                    adversarial_qids.add(rec.get("question_id"))
    except Exception as exc:
        errors.append(f"could not read llm_calls.jsonl: {exc}")
    check(answer_gen_qids >= qid_set,
          f"llm_calls.jsonl missing answer-generation records for: {qid_set - answer_gen_qids}")

    # adversarial extra (only if attempted)
    if os.path.exists(cfg.adversarial_out_path):
        try:
            adv = _load(cfg.adversarial_out_path)
            adv_ids = {a["question_id"] for a in adv}
            check(adversarial_qids >= adv_ids,
                  f"llm_calls.jsonl missing adversarial answer records for: {adv_ids - adversarial_qids}")
        except Exception as exc:
            errors.append(f"invalid adversarial_questions.json: {exc}")

    return (len(errors) == 0), errors


def main():
    cfg = Config()
    ok, errors = validate_all(cfg)
    if ok:
        print("VALIDATION PASSED — all checks green.")
        return 0
    print(f"VALIDATION FAILED — {len(errors)} problem(s):")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
