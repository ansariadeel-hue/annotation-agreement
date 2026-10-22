"""Loading and holding annotation data in long format (item, annotator, label)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = ("item_id", "annotator", "label")


class DataError(ValueError):
    """Raised when annotation data is malformed."""


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def order_categories(labels: Iterable[str]) -> list[str]:
    """Return a stable category order.

    If every label parses as a number the categories are ordered numerically
    (so ``"10"`` comes after ``"9"``); otherwise they are sorted as strings.
    The order matters for weighted kappa and for ordinal/interval alpha.
    """
    unique = set(labels)
    if unique and all(_is_number(v) for v in unique):
        return sorted(unique, key=lambda v: (float(v), v))
    return sorted(unique)


@dataclass
class Ratings:
    """Labels indexed as ``item -> annotator -> label``.

    Missing labels are simply absent from the inner mapping.
    """

    by_item: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_records(cls, records: Iterable[tuple[str, str, str]]) -> "Ratings":
        ratings = cls()
        for item, annotator, label in records:
            ratings.add(item, annotator, label)
        return ratings

    def add(self, item: str, annotator: str, label: str | None) -> None:
        item = str(item).strip()
        annotator = str(annotator).strip()
        if not item or not annotator:
            raise DataError("item_id and annotator must be non-empty")
        if label is None or str(label).strip() == "":
            # An explicit empty label means "not labelled" (missing).
            self.by_item.setdefault(item, {})
            return
        label = str(label).strip()
        row = self.by_item.setdefault(item, {})
        if annotator in row and row[annotator] != label:
            raise DataError(
                f"annotator {annotator!r} gave conflicting labels for item {item!r}: "
                f"{row[annotator]!r} vs {label!r}"
            )
        row[annotator] = label

    @property
    def items(self) -> list[str]:
        return sorted(self.by_item, key=_natural_key)

    @property
    def annotators(self) -> list[str]:
        names = {a for row in self.by_item.values() for a in row}
        return sorted(names, key=_natural_key)

    @property
    def categories(self) -> list[str]:
        return order_categories(l for row in self.by_item.values() for l in row.values())

    @property
    def n_labels(self) -> int:
        return sum(len(row) for row in self.by_item.values())

    def item_labels(self, item: str) -> list[str]:
        return list(self.by_item.get(item, {}).values())

    def paired(self, a: str, b: str) -> tuple[list[str], list[str], list[str]]:
        """Return (items, labels_a, labels_b) for items labelled by both a and b."""
        items, la, lb = [], [], []
        for item in self.items:
            row = self.by_item[item]
            if a in row and b in row:
                items.append(item)
                la.append(row[a])
                lb.append(row[b])
        return items, la, lb


def _natural_key(value: str):
    """Sort key that orders ``item2`` before ``item10``."""
    import re

    parts = re.split(r"(\d+)", value)
    return [(0, int(p)) if p.isdigit() else (1, p) for p in parts]


def load_csv(path: str | Path) -> Ratings:
    """Load a long-format CSV with columns ``item_id, annotator, label``.

    Extra columns are ignored. Empty labels are treated as missing.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise DataError(f"{path}: file is empty")
        header = [h.strip() for h in reader.fieldnames]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise DataError(f"{path}: missing required column(s): {', '.join(missing)}")
        reader.fieldnames = header
        ratings = Ratings()
        for lineno, row in enumerate(reader, start=2):
            try:
                ratings.add(row["item_id"] or "", row["annotator"] or "", row["label"])
            except DataError as exc:
                raise DataError(f"{path}:{lineno}: {exc}") from None
    if not ratings.by_item:
        raise DataError(f"{path}: no rows found")
    return ratings
