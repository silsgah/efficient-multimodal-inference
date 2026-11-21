import pytest

from src.evaluation.metrics import (
    anls_score,
    exact_match_score,
    levenshtein_distance,
    mean_question_score,
    normalized_levenshtein_similarity,
    score_answer,
)


def test_levenshtein_distance_character_edits():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("same", "same") == 0


def test_anls_case_insensitive_and_outer_whitespace_only():
    assert anls_score("  Coca Cola ", ["coca cola"]) == 1.0
    assert anls_score("CocaCola", ["Coca Cola"]) == pytest.approx(8 / 9)


def test_anls_maximum_over_multiple_references():
    assert anls_score("500", ["500.00", "500"]) == 1.0


def test_anls_threshold_is_strict():
    # One edit over two characters is exactly 0.5 normalized distance.
    assert normalized_levenshtein_similarity("ab", "a") == 0.0


def test_anls_empty_answers_are_defined():
    assert anls_score("", [""]) == 1.0
    assert anls_score("answer", [""]) == 0.0


def test_anls_rejects_invalid_reference_input():
    with pytest.raises(ValueError):
        anls_score("answer", [])
    with pytest.raises(TypeError):
        anls_score("answer", "answer")


def test_exact_match_is_not_punctuation_or_number_normalization():
    assert exact_match_score("M.D.", ["m.d."]) == 1.0
    assert exact_match_score("M.D.", ["M. D."]) == 0.0
    assert exact_match_score("500", ["500.00"]) == 0.0


def test_score_answer_returns_both_metrics():
    assert score_answer("500", ["500", "500.00"]) == {
        "exact_match": 1.0,
        "anls": 1.0,
    }


def test_mean_question_score():
    assert mean_question_score([1.0, 0.5, 0.0]) == 0.5
    with pytest.raises(ValueError):
        mean_question_score([])
