"""LLM client + call logger.

The LLM is used ONLY for answer generation. Provider is auto-detected from env:
  1. OPENROUTER_API_KEY -> OpenRouter (OpenAI-compatible HTTP via stdlib urllib; no SDK needed)
  2. ANTHROPIC_API_KEY  -> Anthropic SDK (imported lazily)
  3. otherwise / --mock-llm / on any API error -> deterministic mock

Every call (real or mock) is appended to llm_calls.jsonl with the provider tagged
honestly. The API key is read only from the environment and never logged or persisted.
"""
import datetime
import hashlib
import json
import os
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prompt_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class LLMCallLogger:
    """Appends one JSON object per LLM call to llm_calls.jsonl."""

    def __init__(self, path: str):
        self.path = path

    def log(self, *, stage, question_id, provider, model, p_hash, input_artifacts, output_artifact):
        record = {
            "stage": stage,
            "question_id": question_id,
            "timestamp": _now_iso(),
            "provider": provider,
            "model": model,
            "prompt_hash": p_hash,
            "input_artifacts": input_artifacts,
            "output_artifact": output_artifact,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class LLMClient:
    def __init__(self, config, logger: LLMCallLogger):
        self.config = config
        self.logger = logger
        self._anthropic = None

    def select_provider(self) -> str:
        if self.config.use_mock_llm:
            return "mock"
        if os.environ.get("OPENROUTER_API_KEY"):
            return "openrouter"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "mock"

    def _model_for(self, provider: str) -> str:
        if self.config.model:
            return self.config.model
        if provider == "openrouter":
            return DEFAULT_OPENROUTER_MODEL
        if provider == "anthropic":
            return DEFAULT_ANTHROPIC_MODEL
        return "deterministic-fallback"

    def _call_openrouter(self, model: str, system: str, prompt: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "temperature": 0,
                "max_tokens": self.config.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/deriv-rag-eval-pipeline",
                "X-Title": "deriv-rag-eval-pipeline",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, model: str, system: str, prompt: str) -> str:
        if self._anthropic is None:
            import anthropic  # imported lazily; not needed for the stdlib/mock path
            self._anthropic = anthropic.Anthropic()
        resp = self._anthropic.messages.create(
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def generate(self, *, stage, question_id, system, prompt, input_artifacts, output_artifact, mock_fn):
        """Return (text, provider, model) and log the call. Falls back to mock on any error."""
        p_hash = prompt_hash(system + "\n" + prompt)
        provider = self.select_provider()
        model = self._model_for(provider)
        text = None
        if provider != "mock":
            try:
                if provider == "openrouter":
                    text = self._call_openrouter(model, system, prompt)
                else:
                    text = self._call_anthropic(model, system, prompt)
            except Exception as exc:  # network / SDK / auth -> deterministic fallback
                provider = "mock"
                model = f"deterministic-fallback ({type(exc).__name__})"
                text = mock_fn()
        else:
            model = "deterministic-fallback"
            text = mock_fn()
        self.logger.log(
            stage=stage,
            question_id=question_id,
            provider=provider,
            model=model,
            p_hash=p_hash,
            input_artifacts=input_artifacts,
            output_artifact=output_artifact,
        )
        return text, provider, model
