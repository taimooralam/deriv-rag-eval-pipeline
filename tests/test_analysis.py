"""Tests for deterministic failure-mode and risk-flag assignment (pipeline.analysis)."""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.analysis import analyse
from pipeline.corpus import Question
from pipeline.vocab import FAILURE_MODES, RISK_FLAGS

DOC_SOURCE = {"D1": "policy", "D2": "help_center", "D4": "compliance"}


def _rec(outcome, retrieved):
    return {"question_id": "Q", "retrieved_doc_ids": retrieved, "retrieval_outcome": outcome,
            "top_k_chunks": []}


def _verif(label, present=True, all_in=True, unsupported_numeric=False):
    checks = [{"claim": "n", "supported": not unsupported_numeric,
               "supporting_chunk_ids": [], "method": "numeric_match"}]
    return {"question_id": "Q", "citation_ids_present": present,
            "all_citations_in_retrieval_set": all_in, "support_checks": checks,
            "answer_label": label, "explanation": ""}


def _ans(cited, abstained=False):
    return {"question_id": "Q", "answer": "a", "cited_chunk_ids": cited,
            "abstained": abstained, "notes": ""}


class TestAnalysis(unittest.TestCase):
    def _run(self, answer, verif, rec, cat=""):
        q = Question(question_id="Q", question="q", category_hint=cat, gold_doc_ids=["D1"])
        out = analyse(answer, verif, rec, q, DOC_SOURCE)
        self.assertIn(out["failure_mode"], FAILURE_MODES)
        self.assertIn(out["risk_flag"], RISK_FLAGS)
        self.assertTrue(out["recommended_fix"])
        return out

    def test_miss_is_retrieval_failure(self):
        out = self._run(_ans(["D2::0"]), _verif("not_grounded"), _rec("miss", ["D2"]))
        self.assertEqual(out["failure_mode"], "retrieval_failure")

    def test_citation_out_of_set_is_citation_failure(self):
        out = self._run(_ans(["D9::9"]), _verif("not_grounded", all_in=False),
                        _rec("exact_match", ["D1"]))
        self.assertEqual(out["failure_mode"], "citation_failure")

    def test_compliance_source_not_grounded_is_policy_misread(self):
        out = self._run(_ans(["D4::0"]), _verif("not_grounded"),
                        _rec("exact_match", ["D4"]), cat="compliance")
        self.assertEqual(out["failure_mode"], "policy_misread")
        self.assertEqual(out["risk_flag"], "compliance_risk")

    def test_grounded_is_no_failure_none(self):
        out = self._run(_ans(["D1::0"]), _verif("grounded"), _rec("exact_match", ["D1"]))
        self.assertEqual(out["failure_mode"], "no_failure")
        self.assertEqual(out["risk_flag"], "none")

    def test_abstention_is_no_failure_none(self):
        out = self._run(_ans([], abstained=True), _verif("insufficient_context", present=False),
                        _rec("miss", ["D2"]))
        self.assertEqual(out["failure_mode"], "no_failure")
        self.assertEqual(out["risk_flag"], "none")


if __name__ == "__main__":
    unittest.main()
