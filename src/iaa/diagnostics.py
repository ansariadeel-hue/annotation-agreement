"""QA diagnostics: which items are contested, and which annotators drift."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

from .data import Ratings


@dataclass
class ItemDisagreement:
    item_id: str
    n_labels: int
    majority_label: str | None  # None when the top label is tied
    majority_share: float
    entropy: float  # Shannon entropy of the label distribution, in bits
    label_counts: dict[str, int]
    labels_by_annotator: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def _majority(labels: list[str]) -> tuple[str | None, float]:
    counts = Counter(labels).most_common()
    top_label, top = counts[0]
    tied = len(counts) > 1 and counts[1][1] == top
    return (None if tied else top_label), top / len(labels)


def _entropy(labels: list[str]) -> float:
    n = len(labels)
    return -sum((c / n) * math.log2(c / n) for c in Counter(labels).values()) + 0.0


def disagreement_report(ratings: Ratings, include_unanimous: bool = False) -> list[ItemDisagreement]:
    """Per-item disagreement, most contested items first.

    Items are sorted by ascending majority share, then descending entropy,
    then item id. Items with fewer than two labels are skipped.
    """
    rows = []
    for item in ratings.items:
        by_ann = dict(sorted(ratings.by_item[item].items()))
        labels = list(by_ann.values())
        if len(labels) < 2:
            continue
        majority, share = _majority(labels)
        if share == 1.0 and not include_unanimous:
            continue
        counts = dict(sorted(Counter(labels).items(), key=lambda kv: (-kv[1], kv[0])))
        rows.append(
            ItemDisagreement(item, len(labels), majority, share, _entropy(labels), counts, by_ann)
        )
    order = {item: i for i, item in enumerate(ratings.items)}
    rows.sort(key=lambda r: (r.majority_share, -r.entropy, order[r.item_id]))
    return rows


@dataclass
class AnnotatorScore:
    annotator: str
    n_labelled: int
    n_compared: int  # items where the other annotators had a clear majority
    n_agree: int
    agreement: float | None
    flagged: bool

    def to_dict(self) -> dict:
        return asdict(self)


def annotator_vs_majority(
    ratings: Ratings, threshold: float = 0.7, min_items: int = 5
) -> list[AnnotatorScore]:
    """Agreement of each annotator with the leave-one-out majority.

    For every item an annotator labelled, the majority label is computed from
    the *other* annotators only (so an annotator cannot vote for themselves).
    Items where the others have no strict majority (tie, or fewer than one
    other label) are not counted. An annotator is flagged for retraining when
    their agreement is below ``threshold`` and at least ``min_items`` items
    could be compared. Results are sorted worst first.
    """
    stats: dict[str, list[int]] = {a: [0, 0, 0] for a in ratings.annotators}
    for item in ratings.items:
        row = ratings.by_item[item]
        for ann, label in row.items():
            stats[ann][0] += 1
            others = [l for a, l in row.items() if a != ann]
            if not others:
                continue
            majority, _ = _majority(others)
            if majority is None:
                continue
            stats[ann][1] += 1
            stats[ann][2] += int(label == majority)
    out = []
    for ann, (labelled, compared, agree) in stats.items():
        rate = agree / compared if compared else None
        flagged = rate is not None and compared >= min_items and rate < threshold
        out.append(AnnotatorScore(ann, labelled, compared, agree, rate, flagged))
    out.sort(key=lambda s: (s.agreement if s.agreement is not None else 2.0, s.annotator))
    return out
