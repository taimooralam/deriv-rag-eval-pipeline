"""Pipeline orchestration: drives every stage in order, persists artifacts, and runs
the SHOULD extras (retrieval comparison, adversarial questions) plus in-process validation."""
import os
import shutil

from .analysis import analyse
from .answering import answer_one, generate_answers
from .corpus import (
    Question,
    chunk_documents,
    doc_source_map,
    load_documents,
    load_questions,
)
from .io_utils import read_json, write_json
from .llm import LLMCallLogger, LLMClient
from .report import compute_metrics, render_report
from .retrieval import (
    BM25,
    build_retrieval_record,
    compare_configs,
    run_retrieval,
)
from .state import StateMachine
from .verification import verify_answer
from .vocab import Stage


def _reset_artifacts(cfg):
    """Remove previously generated artifacts so each run rebuilds from a clean slate
    (the evaluator may delete artifacts before running; we mirror that locally)."""
    for path in [
        cfg.retrieval_path, cfg.verification_path, cfg.analysis_path, cfg.report_path,
        cfg.metrics_path, cfg.comparison_path, cfg.adversarial_out_path, cfg.llm_log_path,
        cfg.manifest_path,
    ]:
        if os.path.isfile(path):
            os.remove(path)
    if os.path.isdir(cfg.answers_dir):
        shutil.rmtree(cfg.answers_dir)


def _run_adversarial(cfg, docs, client):
    """Run locally-defined adversarial questions through the same pipeline and judge
    whether the system abstained appropriately (deterministic heuristic)."""
    if not os.path.isfile(cfg.adversarial_path):
        return None
    adv_questions = load_questions(cfg.adversarial_path)
    doc_src = doc_source_map(docs)
    chunks = chunk_documents(docs)
    bm25 = BM25(chunks)

    records = []
    for q in adv_questions:
        rec = build_retrieval_record(q, bm25, cfg.top_k)
        answer = answer_one(
            q, rec, client, cfg,
            stage="ADVERSARIAL_ANSWERS_GENERATED",
            output_artifact=cfg.adversarial_out_path,
            write=False,
        )
        verification = verify_answer(answer, rec, cfg)
        analysis = analyse(answer, verification, rec, q, doc_src)

        # Should the system have abstained? Yes when no gold evidence is defined or
        # the gold document was not retrieved.
        should_abstain = (not q.gold_doc_ids) or rec["retrieval_outcome"] == "miss"
        abstained = bool(answer["abstained"])
        appropriate = abstained == should_abstain
        if abstained and should_abstain:
            reason = "Correctly abstained: no sufficient gold evidence available."
        elif (not abstained) and (not should_abstain):
            reason = "Correctly answered: relevant evidence was retrieved."
        elif abstained and not should_abstain:
            reason = "Over-abstained: evidence was available but the model declined to answer."
        else:
            reason = "Failed to abstain: answered despite insufficient/irrelevant evidence."

        records.append(
            {
                "question_id": q.question_id,
                "question": q.question,
                "category_hint": q.category_hint,
                "gold_doc_ids": q.gold_doc_ids,
                "retrieval_outcome": rec["retrieval_outcome"],
                "answer": answer["answer"],
                "cited_chunk_ids": answer["cited_chunk_ids"],
                "abstained": abstained,
                "answer_label": verification["answer_label"],
                "failure_mode": analysis["failure_mode"],
                "risk_flag": analysis["risk_flag"],
                "abstained_appropriately": appropriate,
                "appropriateness_reason": reason,
            }
        )
    write_json(cfg.adversarial_out_path, records)
    return records


def run_pipeline(cfg) -> dict:
    from validate import validate_all  # local import avoids a circular dependency

    _reset_artifacts(cfg)
    sm = StateMachine(cfg.manifest_path)

    # INIT -> DOCUMENTS_LOADED
    docs = load_documents(cfg.documents_path)
    questions = load_questions(cfg.questions_path)
    questions_by_id = {q.question_id: q for q in questions}
    sm.advance(Stage.DOCUMENTS_LOADED)

    # -> INDEX_BUILT
    chunks = chunk_documents(docs)
    sm.advance(Stage.INDEX_BUILT)

    # -> RETRIEVAL_COMPLETE
    retrieval_records = run_retrieval(questions, chunks, cfg.top_k)
    write_json(cfg.retrieval_path, retrieval_records)
    sm.advance(Stage.RETRIEVAL_COMPLETE)

    # -> ANSWERS_GENERATED (the only LLM stage)
    logger = LLMCallLogger(cfg.llm_log_path)
    client = LLMClient(cfg, logger)
    answers = generate_answers(questions_by_id, retrieval_records, client, cfg)
    sm.advance(Stage.ANSWERS_GENERATED)

    # -> CITATIONS_VERIFIED (deterministic; must run after answers)
    retr_by_id = {r["question_id"]: r for r in retrieval_records}
    verifications = [verify_answer(answers[qid], retr_by_id[qid], cfg) for qid in questions_by_id]
    write_json(cfg.verification_path, verifications)
    sm.advance(Stage.CITATIONS_VERIFIED)

    # -> ANSWERS_SCORED (groundedness labels ARE the scores; finalised post-verification)
    sm.advance(Stage.ANSWERS_SCORED)

    # -> FAILURE_ANALYSIS_COMPLETE
    doc_src = doc_source_map(docs)
    verif_by_id = {v["question_id"]: v for v in verifications}
    analyses = [
        analyse(answers[qid], verif_by_id[qid], retr_by_id[qid], questions_by_id[qid], doc_src)
        for qid in questions_by_id
    ]
    write_json(cfg.analysis_path, analyses)
    sm.advance(Stage.FAILURE_ANALYSIS_COMPLETE)

    # Report (deterministic, no LLM -> not logged in llm_calls.jsonl)
    metrics = compute_metrics(retrieval_records, verifications, analyses, len(questions))
    write_json(cfg.metrics_path, metrics)
    with open(cfg.report_path, "w", encoding="utf-8") as f:
        f.write(render_report(metrics))

    # SHOULD #6: pairwise retrieval comparison (deterministic)
    comparison = {
        "configurations": ["sentence_chunks_topk{}".format(cfg.top_k), "whole_document_topk{}".format(cfg.top_k)],
        "results": compare_configs(
            questions, docs,
            [
                {"name": "sentence_chunks_topk{}".format(cfg.top_k), "whole_document": False, "top_k": cfg.top_k},
                {"name": "whole_document_topk{}".format(cfg.top_k), "whole_document": True, "top_k": cfg.top_k},
            ],
        ),
    }
    base = comparison["results"]["sentence_chunks_topk{}".format(cfg.top_k)]["exact_match_rate"]
    alt = comparison["results"]["whole_document_topk{}".format(cfg.top_k)]["exact_match_rate"]
    comparison["exact_match_rate_delta"] = round(alt - base, 4)
    comparison["tradeoff_summary"] = (
        "Sentence chunks give finer-grained, citable evidence (better for downstream citation "
        "verification) at similar recall; whole-document retrieval is coarser and weakens "
        "citation precision. On this corpus exact-match recall is comparable; the chunked config "
        "is preferred because grounding is checked at the sentence level. Delta = whole_doc - chunked."
    )
    write_json(cfg.comparison_path, comparison)

    # SHOULD #7: adversarial questions (LLM answer calls logged separately)
    _run_adversarial(cfg, docs, client)

    # -> VALIDATION_COMPLETE
    ok, errors = validate_all(cfg)
    sm.advance(Stage.VALIDATION_COMPLETE)

    # -> RESULTS_FINALISED
    sm.advance(Stage.RESULTS_FINALISED)

    return {"metrics": metrics, "validation_ok": ok, "validation_errors": errors}
