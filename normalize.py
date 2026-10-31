#!/usr/bin/env python3
"""
normalize.py - Arabic text normalization for ASR evaluation.

Imports the normalize_arabic function from arabic_data_pipeline.py and wraps
it for convenient use in Phase 2 scripts. Also provides Whisper-compatible
normalization that is applied to both references and hypotheses.
"""
from __future__ import annotations

import re
import sys
import unicodedata
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


def normalize_for_eval(text: str) -> str:
    """
    Normalize text for WER/CER evaluation.

    Applied identically to both references and hypotheses. This is critical:
    Whisper's raw output uses different conventions than Common Voice
    transcripts, so the same normalizer must be applied to both.
    """
    if not isinstance(text, str):
        return ""
    # Apply the Arabic normalization
    t = normalize_arabic(text, DEFAULT_NORM_CFG)
    # Additional cleanup for Whisper outputs
    # Remove any remaining punctuation that normalize_arabic might leave
    t = re.sub(r"[^\u0621-\u064A0-9\s]", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def test_normalize():
    """Quick sanity checks for the normalizer."""
    cases = [
        # (input, expected_is_nonempty)
        ("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ", True),  # with diacritics
        ("هذا اختبار", True),                                 # simple
        ("", False),                                          # empty
        ("   ", False),                                       # whitespace
        ("Hello World", False),                               # Latin only
        ("أإآ → ا", True),                                    # alef normalization
    ]
    for inp, expect_nonempty in cases:
        result = normalize_for_eval(inp)
        assert bool(result) == expect_nonempty, f"Failed: {inp!r} -> {result!r}"
    print("All normalization tests passed.")


if __name__ == "__main__":
    test_normalize()

