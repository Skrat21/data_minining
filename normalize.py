#!/usr/bin/env python3
"""
normalize.py - Arabic text normalization for ASR evaluation.

Imports the normalize_arabic function from arabic_data_pipeline.py and wraps
it for convenient use in Phase 2 scripts. Also provides Whisper-compatible
normalization that is applied to both references and hypotheses.

Changes vs. the original:
  - results are cached (lru_cache). Every text used to be normalized three
    times per evaluation (corpus WER, corpus CER, per-utterance metrics); now
    it is done once.
  - output is IDENTICAL to the original by default, so existing results stay
    comparable.
  - OPTIONAL: spell out digits as Arabic number words before scoring. Whisper
    often outputs "25" where the reference has the number in words, which
    counts as an error. Enable with the environment variable
        EVAL_NUM2WORDS=1          (requires: pip install num2words)
    This CHANGES WER, so if you turn it on, re-evaluate ALL arms with it.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

# Import from the existing pipeline script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_data_pipeline import NormConfig, normalize_arabic

# Default config matching the pipeline (same normalization for all arms)
DEFAULT_NORM_CFG = NormConfig(
    alef=True,
    hamza_carriers=True,
    remove_hamza=True,
    yaa=True,
    ta_marbuta=True,
    keep_latin=False,
)

_NON_ARABIC = re.compile(r"[^\u0621-\u064A0-9\s]")
_SPACES = re.compile(r"\s+")
_DIGIT_RUN = re.compile(r"\d+")

NUM2WORDS = os.environ.get("EVAL_NUM2WORDS", "0") == "1"


def _digits_to_words(text: str) -> str:
    """Replace digit runs by Arabic number words (needs num2words; no-op if missing)."""
    try:
        from num2words import num2words
    except ImportError:
        return text

    def repl(m: re.Match) -> str:
        try:
            return " " + num2words(int(m.group()), lang="ar") + " "
        except Exception:
            return m.group()

    return _DIGIT_RUN.sub(repl, text)


@lru_cache(maxsize=262144)
def _normalize_cached(text: str) -> str:
    # Apply the Arabic normalization
    t = normalize_arabic(text, DEFAULT_NORM_CFG)

    if NUM2WORDS and _DIGIT_RUN.search(t):
        t2 = _digits_to_words(t)
        if t2 != t:
            # re-normalize so the spelled-out words follow the same conventions
            t = normalize_arabic(t2, DEFAULT_NORM_CFG)

    # Remove any remaining punctuation that normalize_arabic might leave
    t = _NON_ARABIC.sub(" ", t)
    # Collapse whitespace
    return _SPACES.sub(" ", t).strip()


def normalize_for_eval(text: str) -> str:
    """
    Normalize text for WER/CER evaluation.

    Applied identically to both references and hypotheses. This is critical:
    Whisper's raw output uses different conventions than Common Voice
    transcripts, so the same normalizer must be applied to both.
    """
    if not isinstance(text, str):
        return ""
    return _normalize_cached(text)


def test_normalize():
    """Quick sanity checks for the normalizer."""
    cases = [
        # (input, expected_is_nonempty)
        ("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ", True),   # with diacritics
        ("هذا اختبار", True),                              # simple
        ("", False),                                       # empty
        ("   ", False),                                    # whitespace
        ("Hello World", False),                            # Latin only
        ("أإآ → ا", True),                                 # alef normalization
    ]
    for inp, expect_nonempty in cases:
        result = normalize_for_eval(inp)
        assert bool(result) == expect_nonempty, f"Failed: {inp!r} -> {result!r}"

    # diacritics must be removed, not turned into word breaks
    assert normalize_for_eval("مُحَمَّد") == normalize_for_eval("محمد")
    # Arabic-Indic and Western digits normalize to the same thing
    assert normalize_for_eval("٢٥") == normalize_for_eval("25")
    # idempotent (refs are already normalized by the data pipeline)
    s = normalize_for_eval("إِنَّ الْعِلْمَ نُورٌ")
    assert normalize_for_eval(s) == s
    print("All normalization tests passed.")


if __name__ == "__main__":
    test_normalize()
