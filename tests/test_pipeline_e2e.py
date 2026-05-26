"""State-machine + full-pipeline integration tests (replayability proof)."""
import os
import shutil
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.config import Config
from pipeline.state import StateMachine
from pipeline.vocab import Stage


class TestStateMachine(unittest.TestCase):
    def test_legal_full_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            sm = StateMachine(os.path.join(d, "m.json"))
            for stage in list(Stage)[1:]:
                sm.advance(stage)
            self.assertEqual(sm.current, Stage.RESULTS_FINALISED)

    def test_illegal_skip_raises(self):
        with tempfile.TemporaryDirectory() as d:
            sm = StateMachine(os.path.join(d, "m.json"))
            with self.assertRaises(RuntimeError):
                sm.advance(Stage.INDEX_BUILT)  # skips DOCUMENTS_LOADED


class TestEndToEnd(unittest.TestCase):
    def test_clean_rebuild_validates_and_is_deterministic(self):
        from pipeline.runner import run_pipeline

        original = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            for name in ("documents.json", "questions.json", "adversarial.json"):
                shutil.copy(os.path.join(REPO_ROOT, name), os.path.join(d, name))
            try:
                os.chdir(d)
                cfg = Config()
                cfg.use_mock_llm = True

                result = run_pipeline(cfg)
                self.assertTrue(result["validation_ok"], result["validation_errors"])

                for artifact in ("retrieval.json", "verification.json", "analysis.json",
                                 "report.md", "metrics.json", "llm_calls.jsonl",
                                 "run_manifest.json", "retrieval_comparison.json",
                                 "adversarial_questions.json"):
                    self.assertTrue(os.path.exists(artifact), f"missing {artifact}")
                self.assertTrue(os.path.isdir("answers"))

                with open("verification.json") as fh:
                    first = fh.read()
                run_pipeline(cfg)  # rebuild from clean
                with open("verification.json") as fh:
                    second = fh.read()
                self.assertEqual(first, second,
                                 "deterministic core must produce identical verification output")
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
