"""Tests for the deterministic citation verifier (pipeline.verification)."""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.config import Config
from pipeline.verification import extract_numeric_claims, verify_answer
from pipeline.vocab import ANSWER_LABELS

CFG = Config()


def _record(chunks):
    return {"question_id": "Q1", "question": "q", "retrieved_doc_ids": ["D1"],
            "top_k_chunks": chunks, "retrieval_outcome": "exact_match"}


def _answer(text, cited, abstained=False):
    return {"question_id": "Q1", "answer": text, "cited_chunk_ids": cited,
            "abstained": abstained, "notes": ""}


class TestVerification(unittest.TestCase):
    def setUp(self):
        self.chunk = {"doc_id": "D1", "chunk_id": "D1::0",
                      "text": "The API permits 120 requests per minute per token.", "score": 1.0}

    def test_numeric_extraction_normalises(self):
        nums = dict(extract_numeric_claims("limit 120 and 10,000 USD over 24 hours"))
        self.assertIn("120", nums)
        self.assertIn("10000", nums)
        self.assertIn("24", nums)

    def test_grounded_when_supported(self):
        rec = _record([self.chunk])
        v = verify_answer(_answer("The API permits 120 requests per minute [D1::0]", ["D1::0"]), rec, CFG)
        self.assertEqual(v["answer_label"], "grounded")
        self.assertIn(v["answer_label"], ANSWER_LABELS)
        self.assertTrue(v["all_citations_in_retrieval_set"])

    def test_not_grounded_when_citation_fabricated(self):
        rec = _record([self.chunk])
        v = verify_answer(_answer("The API permits 120 rpm [D9::9]", ["D9::9"]), rec, CFG)
        self.assertFalse(v["all_citations_in_retrieval_set"])
        self.assertEqual(v["answer_label"], "not_grounded")

    def test_unsupported_number_lowers_label(self):
        rec = _record([self.chunk])
        v = verify_answer(_answer("The API permits 500 requests per minute [D1::0]", ["D1::0"]), rec, CFG)
        self.assertIn(v["answer_label"], ("partially_grounded", "not_grounded"))
        numeric = [sc for sc in v["support_checks"] if sc["method"] == "numeric_match"]
        self.assertTrue(any(not sc["supported"] for sc in numeric))

    def test_abstain_is_insufficient_context(self):
        rec = _record([self.chunk])
        v = verify_answer(_answer("", [], abstained=True), rec, CFG)
        self.assertEqual(v["answer_label"], "insufficient_context")

    def test_missing_citations_not_grounded(self):
        rec = _record([self.chunk])
        v = verify_answer(_answer("Some answer with no citations", []), rec, CFG)
        self.assertEqual(v["answer_label"], "not_grounded")


if __name__ == "__main__":
    unittest.main()
