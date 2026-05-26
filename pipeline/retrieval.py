"""Pure-Python BM25 retrieval + deterministic retrieval-outcome scoring."""
import math
import re
from collections import Counter
from typing import List

from .corpus import Chunk, Question, chunk_documents

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


class BM25:
    """Okapi BM25 over a fixed list of chunks. Title is folded into the indexed text."""

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(f"{c.text} {c.title}") for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.N = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] += 1
        self.df = df

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def _score(self, query_tokens: List[str], idx: int) -> float:
        tf = self.tf[idx]
        dl = self.doc_len[idx]
        score = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1))
            if denom:
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
        return score

    def rank(self, query: str, top_k: int):
        qt = tokenize(query)
        scored = [(self._score(qt, i), self.chunks[i]) for i in range(self.N)]
        # Deterministic ordering: highest score first, ties broken by chunk_id.
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        return scored[:top_k]


def retrieval_outcome(retrieved_doc_ids, gold_doc_ids) -> str:
    """Deterministic comparison of retrieved docs against gold. Never delegated to the LLM."""
    retrieved = set(retrieved_doc_ids)
    gold = set(gold_doc_ids)
    if not gold:
        return "miss"  # no gold defined -> not satisfiable; conservative.
    if gold.issubset(retrieved):
        return "exact_match"
    if gold & retrieved:
        return "partial_match"
    return "miss"


def build_retrieval_record(question: Question, bm25: BM25, top_k: int) -> dict:
    ranked = bm25.rank(question.question, top_k)
    top_k_chunks = []
    retrieved_doc_ids = []
    for score, ch in ranked:
        top_k_chunks.append(
            {
                "doc_id": ch.doc_id,
                "chunk_id": ch.chunk_id,
                "text": ch.text,
                "score": round(float(score), 4),
            }
        )
        if ch.doc_id not in retrieved_doc_ids:
            retrieved_doc_ids.append(ch.doc_id)
    return {
        "question_id": question.question_id,
        "question": question.question,
        "retrieved_doc_ids": retrieved_doc_ids,
        "top_k_chunks": top_k_chunks,
        "retrieval_outcome": retrieval_outcome(retrieved_doc_ids, question.gold_doc_ids),
    }


def run_retrieval(questions: List[Question], chunks: List[Chunk], top_k: int) -> List[dict]:
    bm25 = BM25(chunks)
    return [build_retrieval_record(q, bm25, top_k) for q in questions]


def compare_configs(questions: List[Question], docs, configs: List[dict]) -> dict:
    """Run several retrieval configs and report per-config exact-match rate + per-question docs."""
    out = {}
    n = len(questions) or 1
    for cfg in configs:
        chunks = chunk_documents(docs, whole_document=cfg["whole_document"])
        bm25 = BM25(chunks)
        records = []
        exact = 0
        for q in questions:
            r = build_retrieval_record(q, bm25, cfg["top_k"])
            records.append(
                {
                    "question_id": q.question_id,
                    "retrieved_doc_ids": r["retrieved_doc_ids"],
                    "retrieval_outcome": r["retrieval_outcome"],
                }
            )
            if r["retrieval_outcome"] == "exact_match":
                exact += 1
        out[cfg["name"]] = {
            "config": {"granularity": "whole_document" if cfg["whole_document"] else "sentence_chunks",
                        "top_k": cfg["top_k"]},
            "per_question": records,
            "exact_match_rate": round(exact / n, 4),
        }
    return out
