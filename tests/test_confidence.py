from system_one.confidence import choice_confidence, noul_confidence, score_confidence


def test_choice_confidence_matches_public_formula():
    probabilities = [0.9, 0.06, 0.04]
    expected = (0.9 - 1 / 3) / (1 - 1 / 3)
    assert abs(choice_confidence(probabilities) - expected) < 1e-12


def test_choice_confidence_extremes():
    assert choice_confidence([1 / 3, 1 / 3, 1 / 3]) == 0
    assert choice_confidence([1.0, 0.0, 0.0]) == 1


def test_score_confidence_matches_documented_cases():
    adjacent = score_confidence([0.0, 0.5, 0.5])
    opposite = score_confidence([0.5, 0.0, 0.5])
    documented = score_confidence([0.0, 0.57, 0.43])
    assert abs(adjacent - 0.25) < 1e-12
    assert opposite == 0
    assert abs(documented - (1 - 0.43 / (2 / 3))) < 1e-12


def test_noul_confidence_is_distance_from_half():
    assert noul_confidence(0.5) == 0
    assert noul_confidence(1) == 1
    assert noul_confidence(0) == 1
