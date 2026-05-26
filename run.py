#!/usr/bin/env python3
"""Entry point: rebuild the full pipeline from the input files and produce all artifacts.

Usage:
  python3 run.py                # uses real LLM if OPENROUTER_API_KEY/ANTHROPIC_API_KEY set, else mock
  python3 run.py --mock-llm     # force the deterministic mock LLM (no network/key needed)
  python3 run.py --top-k 4
"""
import argparse
import sys

from pipeline.config import Config
from pipeline.io_utils import load_dotenv
from pipeline.runner import run_pipeline


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description="Replayable RAG evaluation pipeline.")
    parser.add_argument("--mock-llm", action="store_true", help="Force deterministic mock LLM.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    args = parser.parse_args(argv)

    cfg = Config()
    if args.mock_llm:
        cfg.use_mock_llm = True
    if args.top_k is not None:
        cfg.top_k = args.top_k

    result = run_pipeline(cfg)
    m = result["metrics"]
    print("Pipeline complete.")
    print(f"  questions processed     : {m['n_questions']}")
    print(f"  retrieval hit-rate (exact): {m['retrieval_hit_rate_exact']:.0%}")
    print(f"  groundedness labels     : {m['answer_label_counts']}")
    print(f"  failure modes           : {m['failure_mode_counts']}")
    print(f"  risk flags              : {m['risk_flag_counts']}")
    print(f"  validation              : {'PASS' if result['validation_ok'] else 'FAIL'}")
    if not result["validation_ok"]:
        for e in result["validation_errors"]:
            print(f"    - {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
