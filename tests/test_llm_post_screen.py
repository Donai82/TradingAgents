"""Jev screening over the run's own LLM.

Covers the two things that can go wrong in a batched classification: the model
failing to line its verdicts up with the posts, and an over-eager drop. The
rest (which stance counts, how the note reads) is shared with the TypeSafe
backend and covered in test_post_screen.py.
"""

import pytest

from tradingagents.agents import post_screen
from tradingagents.agents.llm_post_screen import (
    PostScreen,
    PostScreenBatch,
    ask_llm,
    screen_with_llm,
)


class _Structured:
    """A stand-in for llm.with_structured_output that returns canned verdicts."""

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _LLM:
    def __init__(self, result):
        self.structured = _Structured(result)

    def with_structured_output(self, schema):
        assert schema is PostScreenBatch
        return self.structured


def _verdict(number, about=0.9, stance="bullish", confidence=0.9):
    return PostScreen(number=number, about=about, stance=stance, confidence=confidence)


@pytest.fixture(autouse=True)
def no_typesafe_key(monkeypatch):
    """These tests are about the LLM backend, so the TypeSafe path stays off."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


# --- backend selection -----------------------------------------------------

def test_a_typesafe_key_takes_precedence_over_the_llm(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(post_screen, "_typesafe_screen", lambda ticker: "typesafe")
    assert post_screen.post_screen("NVDA", llm=object()) == "typesafe"


def test_without_a_key_the_llm_screens(monkeypatch):
    from tradingagents.agents import llm_post_screen

    monkeypatch.setattr(
        llm_post_screen, "screen_with_llm", lambda llm, instrument: (llm, instrument)
    )
    llm = object()
    instrument = post_screen.post_screen("NVDA", llm=llm)
    assert instrument[0] is llm
    # The company name is resolved from a (cached) lookup, so assert on the
    # ticker, which is always part of the label.
    assert "NVDA" in instrument[1]


def test_nothing_to_screen_with_neither_backend():
    assert post_screen.post_screen("NVDA", llm=None) is None


# --- batching --------------------------------------------------------------

def test_one_request_covers_the_whole_batch():
    llm = _LLM(PostScreenBatch(posts=[_verdict(1), _verdict(2), _verdict(3)]))

    results = ask_llm(llm, "NVIDIA (NVDA)", ["a", "b", "c"])

    assert len(results) == 3
    assert len(llm.structured.prompts) == 1, "screening must not cost a call per post"


def test_the_prompt_numbers_the_posts_and_names_the_instrument():
    llm = _LLM(PostScreenBatch(posts=[_verdict(1), _verdict(2)]))

    ask_llm(llm, "NVIDIA (NVDA)", ["first post", "second post"])

    prompt = llm.structured.prompts[0]
    assert "[1] first post" in prompt
    assert "[2] second post" in prompt
    assert "NVIDIA (NVDA)" in prompt


def test_no_posts_asks_nothing():
    llm = _LLM(PostScreenBatch(posts=[]))
    assert ask_llm(llm, "NVIDIA (NVDA)", []) == []
    assert llm.structured.prompts == []


def test_a_very_long_post_is_truncated():
    llm = _LLM(PostScreenBatch(posts=[_verdict(1)]))

    ask_llm(llm, "NVIDIA (NVDA)", ["x" * 5000])

    prompt = llm.structured.prompts[0]
    assert "x" * 600 in prompt, "the post should be cut at the truncation limit"
    assert "x" * 601 not in prompt, "the rest of the post should not be sent"


# --- lining the verdicts up with the posts ---------------------------------

def test_verdicts_are_matched_by_number_not_by_position():
    # Out of order on purpose: a reordered list is still a valid answer.
    llm = _LLM(PostScreenBatch(posts=[_verdict(2, stance="bearish"), _verdict(1)]))

    results = ask_llm(llm, "NVIDIA (NVDA)", ["a", "b"])

    assert [r.number for r in results] == [1, 2]
    assert results[1].stance == "bearish"


@pytest.mark.parametrize(
    "verdicts",
    [
        pytest.param([_verdict(1)], id="too-few"),
        pytest.param([_verdict(1), _verdict(1)], id="duplicate-number"),
        pytest.param([_verdict(1), _verdict(3)], id="out-of-range-number"),
        pytest.param([], id="none"),
    ],
)
def test_a_batch_that_does_not_line_up_is_refused(verdicts):
    # Misreading an off-topic verdict as another post's would silently drop
    # real data, so an unusable batch keeps every post instead.
    assert ask_llm(_LLM(PostScreenBatch(posts=verdicts)), "NVIDIA (NVDA)", ["a", "b"]) is None


def test_a_model_failure_is_refused():
    assert ask_llm(_LLM(RuntimeError("upstream 500")), "NVIDIA (NVDA)", ["a"]) is None


def test_a_model_without_structured_output_is_refused():
    class _NoStructured:
        def with_structured_output(self, *a, **k):
            raise NotImplementedError

    assert ask_llm(_NoStructured(), "NVIDIA (NVDA)", ["a"]) is None


# --- the screen the fetchers consume ---------------------------------------

def test_clear_off_topic_posts_are_dropped_and_stances_counted():
    llm = _LLM(PostScreenBatch(posts=[
        _verdict(1, about=0.05),                 # dropped
        _verdict(2, stance="bullish"),
        _verdict(3, stance="bearish"),
    ]))

    keep, note = screen_with_llm(llm, "NVIDIA (NVDA)")(["a", "b", "c"])

    assert keep == [False, True, True]
    assert "2 of the 3 posts fetched are about NVIDIA (NVDA)" in note
    assert "1 bullish, 1 bearish" in note


def test_an_unsure_stance_reads_as_unclear():
    llm = _LLM(PostScreenBatch(posts=[_verdict(1, stance="bullish", confidence=0.2)]))

    keep, note = screen_with_llm(llm, "NVIDIA (NVDA)")(["a"])

    assert keep == [True]
    assert "1 unclear" in note


def test_a_post_sitting_on_the_threshold_is_kept():
    # Only a clear "not about it" drops a post.
    llm = _LLM(PostScreenBatch(posts=[_verdict(1, about=0.3)]))

    keep, _ = screen_with_llm(llm, "NVIDIA (NVDA)")(["a"])

    assert keep == [True]


def test_an_unusable_batch_leaves_every_post_unscreened():
    llm = _LLM(PostScreenBatch(posts=[_verdict(1)]))

    keep, note = screen_with_llm(llm, "NVIDIA (NVDA)")(["a", "b"])

    assert keep == [True, True]
    assert "screening unavailable" in note


def test_an_empty_source_is_screened_without_asking_the_model():
    llm = _LLM(PostScreenBatch(posts=[]))

    keep, note = screen_with_llm(llm, "NVIDIA (NVDA)")([])

    assert keep == []
    assert llm.structured.prompts == []
    assert note
