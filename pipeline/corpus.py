"""Load and validate the input corpus/questions, and chunk documents for retrieval."""
import json
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Document:
    doc_id: str
    title: str
    source_type: str
    text: str


@dataclass
class Question:
    question_id: str
    question: str
    category_hint: str = ""
    gold_doc_ids: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_type: str
    text: str


# Sentence boundary: split after . ! ? followed by whitespace.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_documents(path) -> List[Document]:
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array of documents")
    docs = []
    seen = set()
    for d in raw:
        doc_id = d["doc_id"]  # required by schema
        if doc_id in seen:
            raise ValueError(f"Duplicate doc_id {doc_id}")
        seen.add(doc_id)
        docs.append(
            Document(
                doc_id=doc_id,
                title=d.get("title", ""),
                source_type=d.get("source_type", "unknown"),
                text=d.get("text", ""),
            )
        )
    return docs


def load_questions(path) -> List[Question]:
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array of questions")
    qs = []
    for q in raw:
        qs.append(
            Question(
                question_id=q["question_id"],  # required by schema
                question=q.get("question", ""),
                category_hint=q.get("category_hint", ""),
                gold_doc_ids=list(q.get("gold_doc_ids", [])),
            )
        )
    return qs


def chunk_documents(docs: List[Document], whole_document: bool = False) -> List[Chunk]:
    """Sentence-level chunks by default; whole-document chunks for the comparison config."""
    chunks: List[Chunk] = []
    for d in docs:
        text = (d.text or "").strip()
        if whole_document:
            chunks.append(Chunk(f"{d.doc_id}::full", d.doc_id, d.title, d.source_type, text))
            continue
        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
        if not sentences:
            sentences = [text] if text else []
        for i, s in enumerate(sentences):
            chunks.append(Chunk(f"{d.doc_id}::{i}", d.doc_id, d.title, d.source_type, s))
    return chunks


def doc_source_map(docs: List[Document]) -> dict:
    return {d.doc_id: d.source_type for d in docs}
