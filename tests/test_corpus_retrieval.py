"""Tests for corpus loading/chunking and BM25 retrieval (pipeline.corpus, pipeline.retrieval)."""
import json
import os
import re
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.corpus import chunk_documents, load_documents, load_questions
from pipeline.retrieval import (
    BM25,
    build_retrieval_record,
    compare_configs,
    retrieval_outcome,
    tokenize,
)
from pipeline.vocab import RETRIEVAL_OUTCOMES

DOCS_PATH = os.path.join(REPO_ROOT, "documents.json")
QS_PATH = os.path.join(REPO_ROOT, "questions.json")


class TestCorpus(unittest.TestCase):
    def test_load_and_chunk_ids_wellformed(self):
        docs = load_documents(DOCS_PATH)
        self.assertTrue(docs)
        chunks = chunk_documents(docs)
        for c in chunks:
            self.assertRegex(c.chunk_id, r"^[A-Za-z0-9_]+::\d+$")
        whole = chunk_documents(docs, whole_document=True)
        self.assertEqual(len(whole), len(docs))
        for c in whole:
            self.assertTrue(c.chunk_id.endswith("::full"))

    def test_load_rejects_non_list_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.json")
            with open(bad, "w") as f:
                json.dump({"not": "a list"}, f)
            with self.assertRaises(ValueError):
                load_documents(bad)
            dup = os.path.join(d, "dup.json")
            with open(dup, "w") as f:
                json.dump([{"doc_id": "D1", "text": "a"}, {"doc_id": "D1", "text": "b"}], f)
            with self.assertRaises(ValueError):
                load_documents(dup)


class TestRetrieval(unittest.TestCase):
    def test_tokenize_keeps_codes(self):
        self.assertEqual(tokenize("HTTP 429 and 2FA!"), ["http", "429", "and", "2fa"])

    def test_bm25_retrieves_gold_for_every_sample_question(self):
        docs = load_documents(DOCS_PATH)
        questions = load_questions(QS_PATH)
        bm25 = BM25(chunk_documents(docs))
        for q in questions:
            top = bm25.rank(q.question, 1)[0][1]
            self.assertIn(top.doc_id, q.gold_doc_ids,
                          f"{q.question_id}: top doc {top.doc_id} not in gold {q.gold_doc_ids}")

    def test_retrieval_outcome_rules(self):
        self.assertEqual(retrieval_outcome(["D1", "D2"], ["D1"]), "exact_match")
        self.assertEqual(retrieval_outcome(["D1"], ["D1", "D3"]), "partial_match")
        self.assertEqual(retrieval_outcome(["D2"], ["D1"]), "miss")
        self.assertEqual(retrieval_outcome(["D1"], []), "miss")

    def test_build_record_dedup_and_vocab(self):
        docs = load_documents(DOCS_PATH)
        questions = load_questions(QS_PATH)
        bm25 = BM25(chunk_documents(docs))
        rec = build_retrieval_record(questions[0], bm25, 5)
        self.assertEqual(len(rec["retrieved_doc_ids"]), len(set(rec["retrieved_doc_ids"])))
        self.assertIn(rec["retrieval_outcome"], RETRIEVAL_OUTCOMES)

    def test_compare_configs_rates_in_range(self):
        docs = load_documents(DOCS_PATH)
        questions = load_questions(QS_PATH)
        out = compare_configs(
            questions, docs,
            [{"name": "a", "whole_document": False, "top_k": 3},
             {"name": "b", "whole_document": True, "top_k": 3}],
        )
        for cfg in out.values():
            self.assertGreaterEqual(cfg["exact_match_rate"], 0.0)
            self.assertLessEqual(cfg["exact_match_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
