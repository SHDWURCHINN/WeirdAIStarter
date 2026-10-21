"""
Evaluation helpers for Weird AI parody outputs.

Lesson 10 connects the verifier idea from reasoning-model evaluation to the
Weird AI project. Instead of checking whether a math answer is symbolically
correct, these functions check measurable features of generated parody lyrics:

- Can we extract the final parody from a longer model response?
- Can we normalize the lyrics into clean lines?
- Do adjacent lines rhyme?
- Are the line count and syllable counts reasonably consistent?

These scores are not a complete measure of creativity or humor. They are simple,
deterministic checks that make automated evaluation possible.
"""

from __future__ import annotations

from typing import Any

from weird_ai.rhyme_tools import analyze_lyrics


FINAL_PARODY_MARKER = "Final Parody:"


def extract_final_parody(text: str) -> str:
    """
    Extract the final parody section from a model response.

    Reasoning-style prompts often produce planning text before the final answer.
    For evaluation, we usually want to score only the final generated lyrics.

    Args:
        text: Full model response text.

    Returns:
        Text after the final "Final Parody:" marker. If the marker is missing,
        return the original text with leading/trailing whitespace removed.
    """
    if not text or not text.strip():
        return ""

    if FINAL_PARODY_MARKER in text:
        # Get everything after the last occurrence of the marker
        return text.rsplit(FINAL_PARODY_MARKER, 1)[1].strip()

    return text.strip()


def normalize_lyrics(text: str) -> list[str]:
    """
    Normalize raw lyric text into clean lyric lines.

    Args:
        text: Raw lyric text.

    Returns:
        A list of non-empty lines with surrounding whitespace removed.
    """
    if not text:
        return []

    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return cleaned_lines


def calculate_rhyme_score(text: str) -> float:
    """
    Calculate the adjacent-line rhyme score for a parody.

    This function should delegate the rhyme work to rhyme_tools.analyze_lyrics.

    Args:
        text: Lyric text to evaluate.

    Returns:
        A float from 0.0 to 1.0.
    """
    extracted = extract_final_parody(text)
    analysis = analyze_lyrics(extracted)
    return float(analysis.get("pairwise_rhyme_score", 0.0))


def calculate_structure_score(text: str, target_line_count: int = 4) -> float:
    """
    Score whether a parody has a reasonable structure.

    This score combines:
    - line count similarity to the target line count
    - syllable consistency across lines

    Args:
        text: Lyric text to evaluate.
        target_line_count: Desired number of non-empty lyric lines.

    Returns:
        A float from 0.0 to 1.0.
    """
    extracted = extract_final_parody(text)
    lines = normalize_lyrics(extracted)

    if not lines:
        return 0.0

    analysis = analyze_lyrics(extracted)
    syllable_consistency = float(analysis.get("syllable_consistency_score", 0.0))

    actual_line_count = len(lines)
    line_count_score = 1.0 - min(
        1.0, abs(actual_line_count - target_line_count) / target_line_count
    )

    raw_score = (line_count_score + syllable_consistency) / 2.0
    return float(max(0.0, min(1.0, raw_score)))


def evaluate_parody(text: str, target_line_count: int = 4) -> dict[str, Any]:
    """
    Run the full Weird AI parody evaluation pipeline.

    Args:
        text: Full model response or raw lyric text.
        target_line_count: Desired number of lyric lines.

    Returns:
        A dictionary containing extracted lyrics, analysis details, component
        scores, an overall score, and a simple pass/fail recommendation.
    """
    extracted_text = extract_final_parody(text)
    lines = normalize_lyrics(extracted_text)
    analysis = analyze_lyrics(extracted_text)

    rhyme_score = calculate_rhyme_score(text)
    structure_score = calculate_structure_score(
        text, target_line_count=target_line_count
    )

    overall_score = (rhyme_score + structure_score) / 2.0
    passed = overall_score >= 0.60 and len(lines) > 0

    return {
        "extracted_text": extracted_text,
        "lines": lines,
        "line_count": len(lines),
        "rhyme_scheme": analysis.get("rhyme_scheme", []),
        "syllable_counts": analysis.get("syllable_counts", []),
        "rhyme_score": rhyme_score,
        "structure_score": structure_score,
        "overall_score": overall_score,
        "passed": passed,
    }