"""Offline general-knowledge base: lead paragraphs of Simple English Wikipedia, retrieved with static embeddings.

Definition questions ("what is photosynthesis", "who was Gandhi") are answered by reading the article lead aloud;
other factual questions pass the best-matching lead to the LLM as context. Built once by scripts/build_kb.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from model2vec import StaticModel
from rapidfuzz import fuzz

from pv.config import model

DEFINITION = [
    re.compile(r"^(?:what|who)\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?(?P<t>[^?]+?)\s*\??$"),
    re.compile(r"^(?:what|who)'s\s+(?:an?\s+|the\s+)?(?P<t>[^?]+?)\s*\??$"),
    re.compile(r"^(?:tell me about|tell me more about|explain|define|describe|what do you know about|give me information (?:on|about))\s+(?:an?\s+|the\s+)?(?P<t>[^?]+?)\s*\??$"),
    re.compile(r"^what does\s+(?P<t>[^?]+?)\s+mean\s*\??$"),
]
# "what is the capital of japan" is a fact about a topic, not a definition of the topic
NOT_DEFINITION = re.compile(r"\b(of|in|on|for|between|from|by|to|with|at)\b|\b(time|date|day|weather|price|score|temperature|meaning)\b")


@dataclass
class Hit:
    title: str
    lead: str
    score: float
    title_match: bool

    def snippet(self, max_words: int = 45, sentences: int = 2) -> str:
        parts = re.split(r"(?<=[.!?])\s+", self.lead.strip())
        out: list[str] = []
        for s_ in parts[:sentences]:
            if len(" ".join(out + [s_]).split()) > max_words and out:
                break
            out.append(s_)
        text = " ".join(out)
        words = text.split()
        return text if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",;:") + "."


def clean_query(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", q.lower())).strip()


def topic_of(question: str) -> tuple[str, bool]:
    """(topic, is_definition_question)."""
    q = question.lower().strip()
    for pat in DEFINITION:
        if m := pat.match(q.rstrip(".! ")):
            t = clean_query(m["t"])
            return t, not NOT_DEFINITION.search(t) and 0 < len(t.split()) <= 5
    return clean_query(q), False


class Knowledge:
    def __init__(self, rag_threshold: float = 0.5) -> None:
        d = model("kb")
        self.emb = np.load(d / "emb.npy", mmap_mode="r")
        self.titles: list[str] = json.loads((d / "titles.json").read_text(encoding="utf-8"))
        self.leads: list[str] = json.loads((d / "leads.json").read_text(encoding="utf-8"))
        self.by_title = {clean_query(t): i for i, t in enumerate(self.titles)}
        self.model = StaticModel.from_pretrained(str(model("embed", "potion-base-8M")))
        self.rag_threshold = rag_threshold

    def _embed(self, text: str) -> np.ndarray:
        v = self.model.encode([text])[0]
        return (v / max(float(np.linalg.norm(v)), 1e-9)).astype(np.float32)

    def search(self, question: str) -> tuple[Hit | None, bool]:
        """Best article for the question and whether it was a definition-style question."""
        topic, is_def = topic_of(question)
        if is_def:
            for cand in (topic, topic[:-1] if topic.endswith("s") else topic, topic[:-2] if topic.endswith("es") else topic):
                if cand in self.by_title:
                    i = self.by_title[cand]
                    return Hit(self.titles[i], self.leads[i], 1.0, True), True
        q = self._embed(topic if is_def else clean_query(question))
        scores = np.concatenate([self.emb[a:a + 40000].astype(np.float32) @ q for a in range(0, len(self.emb), 40000)])
        i = int(np.argmax(scores))
        return Hit(self.titles[i], self.leads[i], float(scores[i]), False), is_def

    def confident_definition(self, hit: Hit, topic: str) -> bool:
        """Read the lead aloud only when the article really is the thing asked about."""
        return hit.title_match or (fuzz.ratio(topic, clean_query(hit.title)) >= 90 and hit.score >= 0.5)

    def answer(self, question: str) -> tuple[str, Hit | None, str]:
        """('extract', hit, text) for a definition, ('context', hit, text) to help the LLM, or ('none', None, '')."""
        hit, is_def = self.search(question)
        if hit is None:
            return "none", None, ""
        topic, _ = topic_of(question)
        if is_def and self.confident_definition(hit, topic):
            return "extract", hit, hit.snippet()
        # only trust a passage whose title is actually named in the question (a similar-sounding article misleads the LLM)
        named = fuzz.partial_ratio(clean_query(hit.title), clean_query(question)) >= 92 and len(hit.title) >= 4
        if named and hit.score >= self.rag_threshold:
            return "context", hit, hit.snippet(max_words=60, sentences=3)
        return "none", hit, ""
