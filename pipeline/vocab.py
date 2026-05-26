"""Controlled vocabularies and pipeline stages, defined once in code and enforced everywhere."""
from enum import Enum


class Stage(str, Enum):
    INIT = "INIT"
    DOCUMENTS_LOADED = "DOCUMENTS_LOADED"
    INDEX_BUILT = "INDEX_BUILT"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    ANSWERS_GENERATED = "ANSWERS_GENERATED"
    CITATIONS_VERIFIED = "CITATIONS_VERIFIED"
    ANSWERS_SCORED = "ANSWERS_SCORED"
    FAILURE_ANALYSIS_COMPLETE = "FAILURE_ANALYSIS_COMPLETE"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


# Canonical ordering used by the state machine to enforce legal transitions.
STAGE_ORDER = list(Stage)


class AnswerLabel(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    NOT_GROUNDED = "not_grounded"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class RetrievalOutcome(str, Enum):
    EXACT_MATCH = "exact_match"
    PARTIAL_MATCH = "partial_match"
    MISS = "miss"


class RiskFlag(str, Enum):
    POLICY_RISK = "policy_risk"
    COMPLIANCE_RISK = "compliance_risk"
    SECURITY_RISK = "security_risk"
    HALLUCINATION_RISK = "hallucination_risk"
    NONE = "none"


class FailureMode(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    CITATION_FAILURE = "citation_failure"
    OVERCLAIM = "overclaim"
    UNSUPPORTED_DETAIL = "unsupported_detail"
    POLICY_MISREAD = "policy_misread"
    NO_FAILURE = "no_failure"


# Frozen string sets used for cheap membership validation in validate.py.
ANSWER_LABELS = frozenset(e.value for e in AnswerLabel)
RETRIEVAL_OUTCOMES = frozenset(e.value for e in RetrievalOutcome)
RISK_FLAGS = frozenset(e.value for e in RiskFlag)
FAILURE_MODES = frozenset(e.value for e in FailureMode)
