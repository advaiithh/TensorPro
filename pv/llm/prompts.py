"""Fixed, short voice prompts (a stable prefix keeps the Ollama KV prompt cache warm)."""
from __future__ import annotations

SYSTEM_FULL = ("You are a voice assistant. Answer the question directly and accurately in one to three short spoken sentences. "
               "No lists, markdown, emoji or code. Spell out numbers naturally. If unsure, say so briefly.")
SYSTEM_LOW = "You are a voice assistant. Answer in one or two short spoken sentences. Plain words only."
SYSTEM_BASELINE = "You are a helpful assistant."
SYSTEM_HINDI = ("You are a voice assistant. Reply in Hindi in one or two short sentences, written in Devanagari. "
                "No lists, markdown or emoji.")

SURVIVAL_REPLY = "I am low on resources right now, so I can only give short answers. Please try a simpler question."


def chars_to_tokens(s: str) -> int:
    return max(1, len(s) // 4)


class History:
    """Rolling last-N turns, capped by an approximate token budget."""

    def __init__(self, max_turns: int = 3) -> None:
        self.max_turns = max_turns
        self.turns: list[tuple[str, str]] = []

    def add(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))
        self.turns = self.turns[-self.max_turns:]

    def clear(self) -> None:
        self.turns = []

    def messages(self, system: str, budget_tokens: int) -> list[dict[str, str]]:
        """System + as many recent turns as fit in budget_tokens (oldest dropped first)."""
        used = chars_to_tokens(system)
        kept: list[tuple[str, str]] = []
        for u, a in reversed(self.turns):
            cost = chars_to_tokens(u) + chars_to_tokens(a)
            if used + cost > budget_tokens:
                break
            kept.insert(0, (u, a))
            used += cost
        msgs = [{"role": "system", "content": system}]
        for u, a in kept:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        return msgs
