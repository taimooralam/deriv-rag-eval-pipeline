"""Pipeline state machine.

Enforces the exact stage ordering from the brief in code, and persists timestamped
transitions to run_manifest.json. validate.py reads the manifest to prove that
CITATIONS_VERIFIED ran after ANSWERS_GENERATED (scoring/analysis cannot precede
deterministic citation verification).
"""
import datetime
import json

from .vocab import STAGE_ORDER, Stage


class StateMachine:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.current = Stage.INIT
        self.transitions = []
        self._record(Stage.INIT)
        self.flush()

    def _record(self, stage: Stage):
        self.transitions.append(
            {
                "stage": stage.value,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

    def advance(self, stage: Stage):
        expected_index = STAGE_ORDER.index(self.current) + 1
        target_index = STAGE_ORDER.index(stage)
        if target_index != expected_index:
            raise RuntimeError(
                f"Illegal stage transition: {self.current.value} -> {stage.value} "
                f"(expected {STAGE_ORDER[expected_index].value})"
            )
        self.current = stage
        self._record(stage)
        self.flush()

    def flush(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {"transitions": self.transitions, "current_stage": self.current.value},
                f,
                indent=2,
            )
            f.write("\n")
