"""Tests for the LLM client/logger and answer helpers (pipeline.llm, pipeline.answering)."""
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.answering import mock_answer, parse_answer
from pipeline.config import Config
from pipeline.llm import LLMCallLogger, LLMClient, prompt_hash

_REQUIRED_KEYS = {"stage", "question_id", "timestamp", "provider", "model",
                  "prompt_hash", "input_artifacts", "output_artifact"}


class TestLLM(unittest.TestCase):
    def test_prompt_hash_deterministic(self):
        self.assertTrue(prompt_hash("x").startswith("sha256:"))
        self.assertEqual(prompt_hash("abc"), prompt_hash("abc"))
        self.assertNotEqual(prompt_hash("abc"), prompt_hash("abd"))

    def test_logger_writes_valid_records(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "llm_calls.jsonl")
            LLMCallLogger(path).log(stage="ANSWERS_GENERATED", question_id="Q1", provider="mock",
                                    model="m", p_hash="sha256:x", input_artifacts=["retrieval.json"],
                                    output_artifact="answers/Q1.json")
            with open(path) as fh:
                rec = json.loads(fh.read().strip())
            self.assertEqual(_REQUIRED_KEYS, set(rec))

    def test_mock_provider_when_forced(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config()
            cfg.use_mock_llm = True
            client = LLMClient(cfg, LLMCallLogger(os.path.join(d, "log.jsonl")))
            self.assertEqual(client.select_provider(), "mock")
            text, provider, _ = client.generate(
                stage="ANSWERS_GENERATED", question_id="Q1", system="s", prompt="p",
                input_artifacts=[], output_artifact="answers/Q1.json",
                mock_fn=lambda: '{"answer":"hi","cited_chunk_ids":["D1::0"],"abstained":false,"notes":""}',
            )
            self.assertEqual(provider, "mock")
            self.assertEqual(parse_answer(text)["answer"], "hi")

    def test_mock_answer_abstains_without_evidence(self):
        self.assertTrue(json.loads(mock_answer("q", []))["abstained"])
        nonempty = json.loads(mock_answer("q", [{"chunk_id": "D1::0", "text": "120 rpm"}]))
        self.assertFalse(nonempty["abstained"])
        self.assertEqual(nonempty["cited_chunk_ids"], ["D1::0"])

    def test_parse_answer_robustness(self):
        self.assertEqual(parse_answer('{"a":1}'), {"a": 1})
        self.assertEqual(parse_answer('noise before {"a":1} after')["a"], 1)
        self.assertIsNone(parse_answer("not json at all"))


if __name__ == "__main__":
    unittest.main()
