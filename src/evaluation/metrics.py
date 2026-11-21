"""Reference-aligned answer metrics for document VQA.

The ANLS definition follows the public DocVQA evaluation implementation:
case-insensitive strings, character-level Levenshtein distance, normalization
by the longer string, a strict 0.5 cutoff, and maximum score over references.

No punctuation, number, unit, date, or dotted-initial normalization is applied.
Those transformations must be separately specified and reported if used.
"""
from __future__ import annotations

from collections.abc import Sequence


DEFAULT_ANLS_THRESHOLD = 0.5


def _canonical_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected str, got {type(value).__name__}")
    return value.strip().lower()


def levenshtein_distance(source: str, target: str) -> int:
    """Compute character-level Levenshtein distance using O(min(n, m)) memory."""
    if source == target:
        return 0
    if len(source) < len(target):
        source, target = target, source
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (source_char != target_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def normalized_levenshtein_similarity(
    prediction: str,
    reference: str,
    threshold: float = DEFAULT_ANLS_THRESHOLD,
) -> float:
    """Return thresholded normalized edit similarity for one reference."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in the interval (0, 1].")

    pred = _canonical_text(prediction)
    ref = _canonical_text(reference)

    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    distance = levenshtein_distance(pred, ref)
    normalized_distance = distance / max(len(pred), len(ref))
    return 1.0 - normalized_distance if normalized_distance < threshold else 0.0


def anls_score(
    prediction: str,
    references: Sequence[str],
    threshold: float = DEFAULT_ANLS_THRESHOLD,
) -> float:
    """Return ANLS for one question, maximizing across accepted references."""
    if isinstance(references, str):
        raise TypeError("references must be a sequence of strings, not a string.")
    refs = list(references)
    if not refs:
        raise ValueError("At least one reference answer is required.")
    if any(not isinstance(ref, str) for ref in refs):
        raise TypeError("Every reference answer must be a string.")

    return max(
        normalized_levenshtein_similarity(prediction, ref, threshold)
        for ref in refs
    )


def exact_match_score(prediction: str, references: Sequence[str]) -> float:
    """Case-insensitive, outer-whitespace-trimmed exact match over references."""
    if isinstance(references, str):
        raise TypeError("references must be a sequence of strings, not a string.")
    refs = list(references)
    if not refs:
        raise ValueError("At least one reference answer is required.")
    pred = _canonical_text(prediction)
    return float(any(pred == _canonical_text(ref) for ref in refs))


def score_answer(
    prediction: str,
    references: Sequence[str],
    threshold: float = DEFAULT_ANLS_THRESHOLD,
) -> dict[str, float]:
    """Compute the two primary answer metrics for one question."""
    return {
        "exact_match": exact_match_score(prediction, references),
        "anls": anls_score(prediction, references, threshold),
    }


def mean_question_score(scores: Sequence[float]) -> float:
    """Compute the unweighted mean over questions; repetitions are not questions."""
    if not scores:
        raise ValueError("scores must contain at least one question.")
    return sum(float(score) for score in scores) / len(scores)
