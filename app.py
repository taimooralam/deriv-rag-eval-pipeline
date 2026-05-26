#!/usr/bin/env python3
"""Minimal single-question service interface (STRETCH #8).

Runs retrieval -> answer generation -> deterministic verification for one new question
and prints structured JSON. Reuses the exact same pipeline components as the batch run.

Usage:
  python3 app.py --question "What are the API rate limits?"
  python3 app.py --question "..." --mock-llm
"""
import argparse
import json
import sys

from pipeline.answering import answer_one
from pipeline.config import Config
from pipeline.corpus import (
    Question,
    chunk_documents,
    doc_source_map,
    load_documents,
)
from pipeline.io_utils import load_dotenv
from pipeline.llm import LLMCallLogger, LLMClient
from pipeline.analysis import analyse
from pipeline.retrieval import BM25, build_retrieval_record
from pipeline.verification import verify_answer


def answer_question(question_text, cfg):
    docs = load_documents(cfg.documents_path)
    chunks = chunk_documents(docs)
    bm25 = BM25(chunks)
    q = Question(question_id="QLIVE", question=question_text, category_hint="", gold_doc_ids=[])

    retrieval_record = build_retrieval_record(q, bm25, cfg.top_k)
    logger = LLMCallLogger(cfg.llm_log_path)
    client = LLMClient(cfg, logger)
    answer = answer_one(q, retrieval_record, client, cfg, stage="SERVICE_ANSWER", write=False)
    verification = verify_answer(answer, retrieval_record, cfg)
    analysis = analyse(answer, verification, retrieval_record, q, doc_source_map(docs))

    return {
        "question": question_text,
        "retrieval": {
            "retrieved_doc_ids": retrieval_record["retrieved_doc_ids"],
            # retrieval_outcome is intentionally omitted: it is only meaningful when a
            # question carries gold_doc_ids, which a live service question does not.
            "top_k_chunks": retrieval_record["top_k_chunks"],
        },
        "answer": answer,
        "verification": verification,
        "analysis": {k: analysis[k] for k in ("answer_label", "failure_mode", "risk_flag")},
    }


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description="Single-question RAG service.")
    parser.add_argument("--question", required=True, help="The question to answer.")
    parser.add_argument("--mock-llm", action="store_true", help="Force deterministic mock LLM.")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = Config()
    if args.mock_llm:
        cfg.use_mock_llm = True
    if args.top_k is not None:
        cfg.top_k = args.top_k

    print(json.dumps(answer_question(args.question, cfg), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
