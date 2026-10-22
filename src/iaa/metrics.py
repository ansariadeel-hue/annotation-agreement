"""Agreement coefficients.

All functions are pure-Python and operate either on paired label lists
(two annotators) or on a :class:`~iaa.data.Ratings` object (many annotators,
possibly with missing labels).
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Sequence

from .data import Ratings, order_categories

WEIGHTS = (None, "linear", "quadratic")
LEVELS = ("nominal", "ordinal", "interval", "ratio")


# --------------------------------------------------------------------------- #
# Percent agreement
# --------------------------------------------------------------------------- #
def item_pairwise_agreement(labels: Sequence[str]) -> float:
    """Fraction of annotator pairs on one item that chose the same label."""
    n = len(labels)
    if n < 2:
        raise ValueError("need at least two labels to measure agreement")
    counts = Counter(labels)
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def percent_agreement(ratings: Ratings) -> float:
    """Mean pairwise percent agreement over items with two or more labels.

    For two annotators this is the plain proportion of items on which they agree.
    """
    scores = [
        item_pairwise_agreement(labels)
        for labels in (ratings.item_labels(i) for i in ratings.items)
        if len(labels) >= 2
    ]
    if not scores:
        raise ValueError("no item has two or more labels")
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------- #
# Cohen's kappa (two raters)
# --------------------------------------------------------------------------- #
def _disagreement_weight(i: int, j: int, k: int, weights: str | None) -> float:
    if weights is None:
        return 0.0 if i == j else 1.0
    if k < 2:
        return 0.0
    d = abs(i - j) / (k - 1)
    if weights == "linear":
        return d
    if weights == "quadratic":
        return d * d
    raise ValueError(f"unknown weights {weights!r}; expected one of {WEIGHTS}")


def confusion_matrix(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    categories: Sequence[str] | None = None,
) -> tuple[list[str], list[list[int]]]:
    """Return ``(categories, matrix)`` with rows = annotator A, columns = annotator B."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must have the same length")
    cats = list(categories) if categories is not None else order_categories([*labels_a, *labels_b])
    index = {c: i for i, c in enumerate(cats)}
    for label in (*labels_a, *labels_b):
        if label not in index:
            raise ValueError(f"label {label!r} not in categories {cats}")
    matrix = [[0] * len(cats) for _ in cats]
    for a, b in zip(labels_a, labels_b):
        matrix[index[a]][index[b]] += 1
    return cats, matrix


def cohen_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    weights: str | None = None,
    categories: Sequence[str] | None = None,
) -> float:
    """Cohen's kappa for two annotators, optionally weighted.

    ``weights`` may be ``None`` (unweighted), ``"linear"`` or ``"quadratic"``.
    For weighted kappa the categories are treated as ordered; pass
    ``categories`` explicitly to control the order (numeric labels are ordered
    numerically by default).

    Returns ``nan`` when chance disagreement is zero (e.g. both annotators used
    a single identical category), where kappa is undefined.
    """
    if weights not in WEIGHTS:
        raise ValueError(f"unknown weights {weights!r}; expected one of {WEIGHTS}")
    if not labels_a:
        raise ValueError("need at least one paired label")
    cats, matrix = confusion_matrix(labels_a, labels_b, categories)
    k = len(cats)
    n = len(labels_a)
    row = [sum(r) for r in matrix]
    col = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    observed = expected = 0.0
    for i in range(k):
        for j in range(k):
            w = _disagreement_weight(i, j, k, weights)
            observed += w * matrix[i][j] / n
            expected += w * row[i] * col[j] / (n * n)
    if expected == 0:
        return float("nan")
    return 1.0 - observed / expected


def pairwise_cohen(
    ratings: Ratings, weights: str | None = None, min_shared: int = 1
) -> dict[tuple[str, str], tuple[int, float]]:
    """Cohen's kappa for every annotator pair: ``{(a, b): (n_shared, kappa)}``."""
    cats = ratings.categories
    out: dict[tuple[str, str], tuple[int, float]] = {}
    for a, b in combinations(ratings.annotators, 2):
        _, la, lb = ratings.paired(a, b)
        if len(la) >= min_shared and la:
            out[(a, b)] = (len(la), cohen_kappa(la, lb, weights=weights, categories=cats))
    return out


# --------------------------------------------------------------------------- #
# Fleiss' kappa (many raters, fixed number per item)
# --------------------------------------------------------------------------- #
def fleiss_kappa_from_table(table: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa from an items x categories count table.

    Every row must sum to the same number of raters ``n >= 2``.
    """
    if not table:
        raise ValueError("empty table")
    n = sum(table[0])
    if n < 2:
        raise ValueError("Fleiss' kappa needs at least two raters per item")
    if any(sum(row) != n for row in table):
        raise ValueError("every item must have the same number of ratings")
    N = len(table)
    k = len(table[0])
    p_j = [sum(row[j] for row in table) / (N * n) for j in range(k)]
    P_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in table]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j)
    if P_e == 1:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def fleiss_table(ratings: Ratings, n_raters: int | None = None) -> tuple[list[str], list[str], list[list[int]]]:
    """Build the Fleiss count table.

    Only items with exactly ``n_raters`` labels are used. By default
    ``n_raters`` is the largest number of labels any item received, so items
    with missing labels are dropped.
    Returns ``(items_used, categories, table)``.
    """
    cats = ratings.categories
    if n_raters is None:
        n_raters = max(len(r) for r in ratings.by_item.values())
    index = {c: i for i, c in enumerate(cats)}
    items, table = [], []
    for item in ratings.items:
        labels = ratings.item_labels(item)
        if len(labels) != n_raters:
            continue
        row = [0] * len(cats)
        for label in labels:
            row[index[label]] += 1
        items.append(item)
        table.append(row)
    return items, cats, table


def fleiss_kappa(ratings: Ratings, n_raters: int | None = None) -> float:
    """Fleiss' kappa over items that received exactly ``n_raters`` labels."""
    _, _, table = fleiss_table(ratings, n_raters)
    if not table:
        raise ValueError("no item has the required number of ratings")
    return fleiss_kappa_from_table(table)


# --------------------------------------------------------------------------- #
# Krippendorff's alpha (any number of raters, missing data, several metrics)
# --------------------------------------------------------------------------- #
def _coincidences(units: Sequence[Sequence[str]], cats: list[str]) -> list[list[float]]:
    index = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    o = [[0.0] * k for _ in range(k)]
    for values in units:
        m = len(values)
        if m < 2:
            continue  # unpairable unit
        counts = Counter(values)
        for c, nc in counts.items():
            for d, nd in counts.items():
                pairs = nc * (nc - 1) if c == d else nc * nd
                o[index[c]][index[d]] += pairs / (m - 1)
    return o


def _delta(level: str, cats: list[str], marginals: list[float]):
    if level == "nominal":
        return lambda i, j: 0.0 if i == j else 1.0
    try:
        values = [float(c) for c in cats]
    except ValueError:
        raise ValueError(f"level {level!r} needs numeric labels; got {cats}") from None
    if level == "interval":
        return lambda i, j: (values[i] - values[j]) ** 2
    if level == "ratio":
        def ratio(i, j):
            s = values[i] + values[j]
            return 0.0 if s == 0 else ((values[i] - values[j]) / s) ** 2
        return ratio
    if level == "ordinal":
        def ordinal(i, j):
            if i == j:
                return 0.0
            lo, hi = min(i, j), max(i, j)
            total = sum(marginals[lo : hi + 1]) - (marginals[lo] + marginals[hi]) / 2
            return total * total
        return ordinal
    raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")


def krippendorff_alpha_units(units: Sequence[Sequence[str]], level: str = "nominal") -> float:
    """Krippendorff's alpha from a list of units, each a list of the values it received.

    Missing values are simply omitted from a unit. Units with fewer than two
    values cannot be paired and are ignored.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    units = [list(map(str, u)) for u in units if len(u) >= 2]
    if not units:
        raise ValueError("no unit has two or more values")
    cats = order_categories(v for u in units for v in u)
    o = _coincidences(units, cats)
    k = len(cats)
    n_c = [sum(o[i]) for i in range(k)]
    n = sum(n_c)
    delta = _delta(level, cats, n_c)
    d_o = sum(o[i][j] * delta(i, j) for i in range(k) for j in range(k))
    d_e = sum(n_c[i] * n_c[j] * delta(i, j) for i in range(k) for j in range(k))
    if d_e == 0:
        return float("nan")
    return 1.0 - (n - 1) * d_o / d_e


def krippendorff_alpha(ratings: Ratings, level: str = "nominal") -> float:
    """Krippendorff's alpha for a :class:`Ratings` object (handles missing labels)."""
    return krippendorff_alpha_units([ratings.item_labels(i) for i in ratings.items], level)


def is_nan(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)
