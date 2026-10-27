"""Tokenizer for the lexical (BM25) retriever."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# Small English stopword list. Deliberately excludes words that carry meaning in
# this domain ("attention", "model", "few", "shot", ...).
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each for from
    further had has have having he her here hers herself him himself his how i if in into is
    it its itself just me more most my myself no nor not of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we
    were what when where which while who whom why will with would you your yours yourself
    yourselves also may might must shall via using use used e g i e et al
    """.split()
)


def _stem(token: str) -> str:
    """Minimal plural folding ("transformers" -> "transformer", "queries" -> "query").

    Full stemmers (Porter) over-merge technical terms ("generative"/"generation"),
    so only the safest English plural suffixes are removed.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
