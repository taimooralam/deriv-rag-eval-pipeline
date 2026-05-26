PY ?= python3

.PHONY: run run-mock validate test clean help

help:
	@echo "make run        - rebuild all artifacts (real LLM if a key is set, else deterministic mock)"
	@echo "make run-mock   - rebuild all artifacts with the deterministic mock LLM (no key/network)"
	@echo "make validate   - run all validation checks against the produced artifacts"
	@echo "make test       - run the unit + integration test suite (stdlib unittest, no installs)"
	@echo "make clean      - delete all generated artifacts"

run:
	$(PY) run.py

run-mock:
	$(PY) run.py --mock-llm

validate:
	$(PY) validate.py

test:
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf answers retrieval.json verification.json analysis.json report.md \
		metrics.json llm_calls.jsonl run_manifest.json retrieval_comparison.json \
		adversarial_questions.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
