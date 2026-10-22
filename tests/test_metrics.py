import math

import pytest

from iaa import (
    Ratings,
    cohen_kappa,
    confusion_matrix,
    fleiss_kappa,
    fleiss_kappa_from_table,
    krippendorff_alpha,
    krippendorff_alpha_units,
    pairwise_cohen,
    percent_agreement,
)

# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability":
# 4 observers x 12 units, None = missing value.
KRIPP = {
    "A": [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    "C": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
}

# Fleiss' kappa worked example (10 subjects, 14 raters, 5 categories) as
# reproduced on Wikipedia's "Fleiss' kappa" article: kappa = 0.210.
FLEISS_TABLE = [
    [0, 0, 0, 0, 14],
    [0, 2, 6, 4, 2],
    [0, 0, 3, 5, 6],
    [0, 3, 9, 2, 0],
    [2, 2, 8, 1, 1],
    [7, 7, 0, 0, 0],
    [3, 2, 6, 3, 0],
    [2, 5, 3, 2, 2],
    [6, 5, 2, 1, 0],
    [0, 2, 2, 3, 7],
]


def kripp_ratings() -> Ratings:
    r = Ratings()
    for obs, values in KRIPP.items():
        for u, v in enumerate(values, start=1):
            r.add(f"u{u}", obs, None if v is None else str(v))
    return r


def table_to_ratings(table) -> Ratings:
    r = Ratings()
    for i, row in enumerate(table):
        rater = 0
        for cat, count in enumerate(row):
            for _ in range(count):
                r.add(f"s{i}", f"r{rater}", f"c{cat}")
                rater += 1
    return r


# --- Krippendorff's alpha ---------------------------------------------------
@pytest.mark.parametrize(
    "level, expected",
    [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849), ("ratio", 0.797)],
)
def test_krippendorff_canonical_example(level, expected):
    assert krippendorff_alpha(kripp_ratings(), level) == pytest.approx(expected, abs=5e-4)


def test_krippendorff_two_coder_binary_example():
    # Krippendorff (2011), two observers, binary data, no missing values: alpha = 0.095
    a = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    b = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    units = [[str(x), str(y)] for x, y in zip(a, b)]
    assert krippendorff_alpha_units(units) == pytest.approx(0.095, abs=5e-4)


def test_krippendorff_perfect_agreement_is_one():
    units = [["a", "a", "a"], ["b", "b"], ["c", "c", "c"]]
    assert krippendorff_alpha_units(units) == pytest.approx(1.0)


def test_krippendorff_ignores_unpairable_units():
    units = [["1", "1"], ["2", "2"], ["3"]]  # lone value contributes nothing
    assert krippendorff_alpha_units(units) == pytest.approx(
        krippendorff_alpha_units([["1", "1"], ["2", "2"]])
    )


def test_krippendorff_no_variation_is_nan():
    assert math.isnan(krippendorff_alpha_units([["x", "x"], ["x", "x"]]))


def test_krippendorff_interval_rejects_text_labels():
    with pytest.raises(ValueError):
        krippendorff_alpha_units([["low", "high"], ["low", "low"]], level="interval")


# --- Fleiss' kappa ------------------------------------------------------------
def test_fleiss_reference_table():
    assert fleiss_kappa_from_table(FLEISS_TABLE) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_from_ratings_matches_table():
    assert fleiss_kappa(table_to_ratings(FLEISS_TABLE)) == pytest.approx(
        fleiss_kappa_from_table(FLEISS_TABLE)
    )


def test_fleiss_requires_equal_raters():
    with pytest.raises(ValueError):
        fleiss_kappa_from_table([[2, 0], [1, 2]])


def test_fleiss_uses_only_complete_items():
    r = kripp_ratings()
    # Only the 8 units rated by all four observers enter Fleiss' kappa.
    from iaa.metrics import fleiss_table

    items, _, table = fleiss_table(r)
    assert len(items) == 8 and all(sum(row) == 4 for row in table)


# --- Cohen's kappa -----------------------------------------------------------
def test_cohen_wikipedia_example():
    # 50 proposals: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po 0.7, pe 0.5, kappa 0.4
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_linear_weights_hand_computed():
    # observed weighted disagreement 0.25, expected 7/16 -> kappa = 3/7
    assert cohen_kappa(list("0122"), list("0221"), weights="linear") == pytest.approx(3 / 7)


def test_cohen_quadratic_matches_covariance_identity():
    a = [1, 2, 3, 4, 5, 3, 2, 4, 1, 5, 3, 3]
    b = [1, 3, 3, 5, 4, 2, 2, 4, 2, 5, 3, 4]
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n
    va = sum((x - ma) ** 2 for x in a) / n
    vb = sum((y - mb) ** 2 for y in b) / n
    expected = 2 * cov / (va + vb + (ma - mb) ** 2)
    got = cohen_kappa(list(map(str, a)), list(map(str, b)), weights="quadratic")
    assert got == pytest.approx(expected)


def test_weighted_equals_unweighted_for_two_categories():
    a, b = list("0101100110"), list("0111000110")
    k = cohen_kappa(a, b)
    assert cohen_kappa(a, b, weights="linear") == pytest.approx(k)
    assert cohen_kappa(a, b, weights="quadratic") == pytest.approx(k)


def test_weights_penalise_near_misses_less():
    a = list("1234512345")
    b = list("1234521354")  # every disagreement is off by one
    assert cohen_kappa(a, b, weights="linear") > cohen_kappa(a, b)
    assert cohen_kappa(a, b, weights="quadratic") > cohen_kappa(a, b, weights="linear")


def test_numeric_categories_ordered_numerically():
    # With string sorting "10" < "9" would break weighting; numeric ordering must be used.
    cats, _ = confusion_matrix(["9", "10", "8"], ["10", "9", "8"])
    assert cats == ["8", "9", "10"]


def test_cohen_rejects_unknown_weights():
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a"], weights="cubic")


def test_confusion_matrix_counts():
    cats, m = confusion_matrix(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert cats == ["a", "b"]
    assert m == [[1, 1], [0, 2]]


# --- Percent agreement -------------------------------------------------------
def test_percent_agreement_two_annotators():
    r = Ratings.from_records(
        [("1", "x", "a"), ("1", "y", "a"), ("2", "x", "a"), ("2", "y", "b"),
         ("3", "x", "b"), ("3", "y", "b"), ("4", "x", "c"), ("4", "y", "a")]
    )
    assert percent_agreement(r) == pytest.approx(0.5)


def test_percent_agreement_pairwise_three_annotators():
    # item 1: a,a,b -> 1 of 3 pairs agree; item 2: unanimous -> 1.0
    r = Ratings.from_records(
        [("1", "x", "a"), ("1", "y", "a"), ("1", "z", "b"),
         ("2", "x", "c"), ("2", "y", "c"), ("2", "z", "c")]
    )
    assert percent_agreement(r) == pytest.approx((1 / 3 + 1) / 2)


def test_pairwise_cohen_uses_shared_items_only():
    pairs = pairwise_cohen(kripp_ratings())
    assert pairs[("A", "C")][0] == 8
    assert pairs[("B", "D")][0] == 10
