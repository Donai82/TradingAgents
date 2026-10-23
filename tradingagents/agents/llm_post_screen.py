"""Jev's two post questions, asked of the run's own LLM.

The TypeSafe backend sends each post to System One, which answers the questions
as typed, calibrated probabilities. This module asks the same two questions of
whatever model the run is already configured with — an OpenCode Go
subscription, a ChatGPT sign-in, or an ordinary API key — so screening works
without a second paid service.

The trade-off is calibration. System One returns probabilities trained for the
question; a general model returns its own confidence, which is close but not
identical. The thresholds are therefore the same, generous ones the typed
answers were read with: a post is dropped only on a clear "not about it", and a
stance counts only when the model is reasonably sure. A model that is unsure
yields "unclear", which the Sentiment Analyst is already told how to read.

One request covers a whole batch of posts (structured output, one entry per
post) rather than one per post, which is what makes this cheap enough to run on
every sentiment analysis. Batching is also why the posts are numbered and the
model must echo the number: a batch's results are only usable if every post can
be matched to its own verdict, and an unmatched post is left unscreened.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from tradingagents.agents.structured import bind_structured, invoke_structured

logger = logging.getLogger(__name__)

__all__ = ["PostScreen", "ask_llm", "screen_with_llm"]

# Matches the TypeSafe backend's criteria, restated for a model that has not
# been trained on this question. Dropping a post is the destructive choice, so
# the bar is a clear "no"; a stance is counted only above the same 0.5 line
# System One's confidence was read at.
_OFF_TOPIC_BELOW = 0.3
_STANCE_CONFIDENCE = 0.5

# A post longer than this is truncated. Social posts are short; anything longer
# is a link dump or an article, and the question turns on its opening claim.
_MAX_POST_CHARS = 600

_PROMPT = """\
You are screening social-media posts for a financial research report on \
{instrument}.

For each numbered post, decide two things:

1. `about`: the probability that the post is actually about {instrument} — the \
company, its stock, its products, or its outlook. A post that merely lists the \
ticker among many, is spam or promotion, or discusses a different company is \
not about it.
2. `stance`: what the author expects for the price of {instrument}'s stock.
   - bullish: the author expects it to rise, or is buying or holding long
   - bearish: the author expects it to fall, or is selling or shorting
   - neutral: no view of the author's own (a question, news without opinion, \
or someone else's view quoted)
3. `confidence`: how sure you are of the stance, from 0 to 1. Use a low value \
when the post is ambiguous or the author has no opinion.

Judge the author's own view. Being off-topic is a low `about` score, not a \
neutral stance. Answer with one entry per post, each carrying that post's own \
number, and do not add posts that were not given.
"""


class PostScreen(BaseModel):
    """One post's two answers: relevance, stance, and how sure of the stance."""

    number: int = Field(description="The post's number, copied from the input")
    about: float = Field(ge=0.0, le=1.0, description="Probability the post is about the instrument")
    stance: Literal["bullish", "bearish", "neutral"] = Field(description="The author's view")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the stance, 0 to 1"
    )


class PostScreenBatch(BaseModel):
    """The verdicts for a batch of posts, in the order they were given."""

    posts: list[PostScreen] = Field(description="One entry per post, each echoing its number")


def _truncate(posts: list[str]) -> list[str]:
    return [p[:_MAX_POST_CHARS] for p in posts]


def _numbered(posts: list[str]) -> str:
    return "\n\n".join(
        f"[{i}] {post}" for i, post in enumerate(_truncate(posts), start=1)
    )


def _align(results, posts: list[str]) -> list[PostScreen] | None:
    """Match verdicts to posts by number, or None if the batch is unusable.

    A model that drops, duplicates or renumbers an entry leaves the mapping
    ambiguous, and mislabelling a post as off-topic would silently delete it
    from the report. Rather than guess, the caller keeps every post unscreened.
    """
    by_number: dict[int, PostScreen] = {}
    for result in results:
        if result.number in by_number or not 1 <= result.number <= len(posts):
            return None
        by_number[result.number] = result
    if len(by_number) != len(posts):
        return None
    return [by_number[i] for i in range(1, len(posts) + 1)]


def ask_llm(llm, instrument: str, posts: list[str]):
    """The aligned verdicts for ``posts``, or None if the model did not answer.

    One structured request for the whole batch. Any failure — a model without
    structured output, malformed JSON, a batch that does not line up — returns
    None, which the caller reads as "screening unavailable".
    """
    if not posts:
        return []

    structured = bind_structured(llm, PostScreenBatch, "Jev post screening")
    if structured is None:
        return None

    prompt = f"{_PROMPT.format(instrument=instrument)}\n\nPosts:\n{_numbered(posts)}"
    batch = invoke_structured(structured, prompt, "Jev post screening")
    if batch is None:
        return None

    aligned = _align(batch.posts, posts)
    if aligned is None:
        logger.warning(
            "Jev screening for %s: the model returned %d verdicts for %d posts",
            instrument, len(batch.posts), len(posts),
        )
    return aligned


def screen_with_llm(llm, instrument: str):
    """A screen callable over an LLM, shaped like the TypeSafe one."""
    from .post_screen import _screen_note, _unscreened

    def screen(posts: list[str]) -> tuple[list[bool], str]:
        results = ask_llm(llm, instrument, posts)
        if results is None:
            return _unscreened(instrument, posts, "the model did not answer")
        keep = [r.about >= _OFF_TOPIC_BELOW for r in results]
        stances = [
            r.stance if r.confidence >= _STANCE_CONFIDENCE else "unclear"
            for r, kept in zip(results, keep, strict=True)
            if kept
        ]
        return keep, _screen_note(instrument, len(stances), len(posts), stances)

    return screen
