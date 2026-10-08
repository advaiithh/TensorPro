"""Offline knowledge base: definition routing, grounded extracts, and refusal to trust look-alike articles."""
import pytest

from pv.knowledge.kb import Hit, Knowledge, topic_of


@pytest.fixture(scope="module")
def kb():
    return Knowledge()


def test_topic_extraction():
    assert topic_of("What is photosynthesis?") == ("photosynthesis", True)
    assert topic_of("who was Mahatma Gandhi") == ("mahatma gandhi", True)
    assert topic_of("tell me about the Eiffel Tower") == ("eiffel tower", True)
    assert topic_of("what is the capital of Japan")[1] is False        # a fact about a topic, not a definition
    assert topic_of("why is the sky blue")[1] is False


def test_definitions_are_read_from_the_article(kb):
    for q, title in [("what is photosynthesis", "Photosynthesis"), ("who is Albert Einstein", "Albert Einstein"),
                     ("what is a black hole", "Black hole"), ("tell me about elephants", "Elephant")]:
        kind, hit, text = kb.answer(q)
        assert kind == "extract" and hit.title == title, (q, kind, hit and hit.title)
        assert 5 <= len(text.split()) <= 50


def test_lookalike_articles_are_not_trusted(kb):
    kind, hit, _ = kb.answer("who was the first president of the United States")
    assert kind != "context" or "president" in hit.title.lower() or "united states" in hit.title.lower()
    kind, hit, text = kb.answer("what is diabetes")
    assert not (kind == "extract" and hit.title == "American Diabetes Association")


def test_named_topic_gives_context_for_a_fact_question(kb):
    kind, hit, text = kb.answer("who wrote Romeo and Juliet")
    assert kind == "context" and hit.title == "Romeo and Juliet" and "Shakespeare" in text


def test_lookup_is_fast(kb):
    import time
    kb.answer("what is the capital of Japan")
    t = time.perf_counter()
    for _ in range(20):
        kb.answer("who painted the Mona Lisa")
    assert (time.perf_counter() - t) / 20 < 0.15


def test_snippet_is_short():
    h = Hit("X", "First sentence is here. Second sentence is also here. Third one. " + "word " * 100, 1.0, True)
    assert len(h.snippet().split()) <= 45
