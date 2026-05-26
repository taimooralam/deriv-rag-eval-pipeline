"""Grounded answer generation (the only LLM-touching stage).

One LLM call per question. The model is instructed to answer ONLY from the
provided evidence and to cite supporting chunk IDs. Output is strict JSON;
parsing is defensive. A deterministic mock is used when no real LLM is available.
"""
import json
import os
import re

from .io_utils import ensure_dir, write_json

SYSTEM = (
    "You are a careful support knowledge assistant. Answer the user's question USING ONLY "
    "the provided evidence chunks.\n"
    "Rules:\n"
    "- Use only facts found in the evidence. Do not add outside knowledge.\n"
    "- Cite the supporting chunk id(s) inline in square brackets, e.g. [D1::0], right after "
    "the claim they support.\n"
    "- If the evidence is insufficient to answer, set \"abstained\" to true and explain in \"notes\".\n"
    "- Never advise the user to do anything the evidence prohibits.\n"
    "Return ONLY a JSON object with keys: answer (string), cited_chunk_ids (array of strings), "
    "abstained (boolean), notes (string). No text outside the JSON."
)

_CITE_RE = re.compile(r"\[([A-Za-z0-9_]+::[A-Za-z0-9_]+)\]")


def build_prompt(question: str, chunks) -> str:
    lines = [f"Question: {question}", "", "Evidence chunks:"]
    if chunks:
        for c in chunks:
            lines.append(f"[{c['chunk_id']}] {c['text']}")
    else:
        lines.append("(no evidence retrieved)")
    lines += ["", "Respond with ONLY the JSON object described in the system instructions."]
    return "\n".join(lines)


def mock_answer(question: str, chunks) -> str:
    """Deterministic stand-in for the LLM: synthesises a citation-bearing answer from the
    top retrieved chunks, and abstains when no evidence was retrieved."""
    if not chunks:
        return json.dumps(
            {
                "answer": "",
                "cited_chunk_ids": [],
                "abstained": True,
                "notes": "No evidence retrieved; abstaining (deterministic mock).",
            }
        )
    used = chunks[:2]
    cited = [c["chunk_id"] for c in used]
    snippet = " ".join(c["text"] for c in used)
    inline = "".join(f"[{cid}]" for cid in cited)
    return json.dumps(
        {
            "answer": f"Based on the retrieved policy evidence: {snippet} {inline}",
            "cited_chunk_ids": cited,
            "abstained": False,
            "notes": "Deterministic mock answer synthesised verbatim from top retrieved chunks.",
        }
    )


def parse_answer(text):
    """Parse model JSON; recover from fenced/noisy output; return None on failure."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _normalise(parsed, question_id) -> dict:
    parsed = parsed or {}
    obj = {
        "question_id": question_id,
        "answer": str(parsed.get("answer", "")),
        "cited_chunk_ids": [str(c) for c in parsed.get("cited_chunk_ids", []) if c],
        "abstained": bool(parsed.get("abstained", False)),
        "notes": str(parsed.get("notes", "")),
    }
    # If the model cited inline but left the list empty, recover ids from the text.
    if not obj["cited_chunk_ids"] and not obj["abstained"]:
        inline = list(dict.fromkeys(_CITE_RE.findall(obj["answer"])))
        obj["cited_chunk_ids"] = inline
    return obj


def answer_one(question, retrieval_record, client, cfg, stage="ANSWERS_GENERATED",
               output_artifact=None, write=False) -> dict:
    chunks = retrieval_record["top_k_chunks"]
    prompt = build_prompt(question.question, chunks)
    out_path = output_artifact or os.path.join(cfg.answers_dir, f"{question.question_id}.json")
    text, _provider, _model = client.generate(
        stage=stage,
        question_id=question.question_id,
        system=SYSTEM,
        prompt=prompt,
        input_artifacts=[cfg.retrieval_path],
        output_artifact=out_path,
        mock_fn=lambda: mock_answer(question.question, chunks),
    )
    parsed = parse_answer(text)
    if parsed is None:
        parsed = {"answer": "", "cited_chunk_ids": [], "abstained": True,
                  "notes": "Failed to parse LLM output; abstaining."}
    answer_obj = _normalise(parsed, question.question_id)
    if write:
        ensure_dir(cfg.answers_dir)
        write_json(out_path, answer_obj)
    return answer_obj


def generate_answers(questions_by_id, retrieval_records, client, cfg) -> dict:
    ensure_dir(cfg.answers_dir)
    answers = {}
    for rec in retrieval_records:
        qid = rec["question_id"]
        answers[qid] = answer_one(questions_by_id[qid], rec, client, cfg, write=True)
    return answers
