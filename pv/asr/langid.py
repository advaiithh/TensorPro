"""Cheap language ID: script heuristics on text, Whisper detection on the first ~2 s of audio as fallback."""
from __future__ import annotations

import re

import numpy as np

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINDI_WORDS = {"kya", "hai", "kaise", "aap", "mujhe", "batao", "samay", "kitna", "nahi", "haan", "mera", "tum",
               "kaun", "kahan", "aaj", "kal", "namaste", "dhanyavaad", "shukriya"}


def from_text(text: str) -> str | None:
    """'hi' when the text is Devanagari or clearly romanised Hindi, 'en' for plain English, None if unsure."""
    if not text.strip():
        return None
    letters = re.findall(r"\w", text)
    if letters and len(DEVANAGARI.findall(text)) / len(letters) > 0.3:
        return "hi"
    words = re.findall(r"[a-z']+", text.lower())
    if words and sum(w in HINDI_WORDS for w in words) / len(words) >= 0.3:
        return "hi"
    return "en" if words else None


def from_audio(whisper, pcm16: np.ndarray, allowed: tuple[str, ...] = ("en", "hi")) -> str:
    """Whisper-multilingual detection on the first 2 s, restricted to the languages we ship."""
    lang, _ = whisper.detect_language(pcm16[: 2 * 16000])
    return lang if lang in allowed else "en"
