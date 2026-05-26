"""Central configuration. All paths and tunable knobs live here so a run is reproducible."""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Inputs
    documents_path: str = "documents.json"
    questions_path: str = "questions.json"
    adversarial_path: str = "adversarial.json"

    # Outputs / artifacts
    answers_dir: str = "answers"
    retrieval_path: str = "retrieval.json"
    verification_path: str = "verification.json"
    analysis_path: str = "analysis.json"
    report_path: str = "report.md"
    metrics_path: str = "metrics.json"
    comparison_path: str = "retrieval_comparison.json"
    adversarial_out_path: str = "adversarial_questions.json"
    llm_log_path: str = "llm_calls.jsonl"
    manifest_path: str = "run_manifest.json"

    # Retrieval: number of sentence chunks to retrieve as evidence per question.
    top_k: int = 5

    # LLM. Provider is auto-detected at call time from available env keys
    # (OpenRouter -> Anthropic -> deterministic mock). `model` may be left blank
    # to use a provider-appropriate default; override with the LLM_MODEL env var.
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", ""))
    max_tokens: int = 700
    use_mock_llm: bool = False

    # Verification heuristic thresholds (documented knobs)
    lexical_grounded_threshold: float = 0.35
    lexical_partial_threshold: float = 0.18

    def __post_init__(self):
        if os.environ.get("USE_MOCK_LLM") == "1":
            self.use_mock_llm = True
