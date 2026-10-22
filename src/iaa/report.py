"""Assemble all metrics into one report (dict) and render it as text."""

from __future__ import annotations

import math

from . import metrics
from .data import Ratings
from .diagnostics import annotator_vs_majority, disagreement_report


def landis_koch(value: float | None) -> str:
    """Landis & Koch (1977) verbal band for a kappa-like coefficient."""
    if value is None or math.isnan(value):
        return "undefined"
    if value < 0:
        return "poor"
    for upper, name in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")):
        if value <= upper:
            return name
    return "almost perfect"


def _safe(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
    except ValueError as exc:
        return None, str(exc)
    if isinstance(value, float) and math.isnan(value):
        return None, "undefined (no variation in labels)"
    return value, None


def build_report(
    ratings: Ratings,
    level: str = "nominal",
    weights: str | None = None,
    threshold: float = 0.7,
    min_items: int = 5,
    top: int = 10,
    pair: tuple[str, str] | None = None,
) -> dict:
    annotators = ratings.annotators
    n_cells = len(ratings.items) * len(annotators)
    report: dict = {
        "summary": {
            "items": len(ratings.items),
            "annotators": len(annotators),
            "labels": ratings.n_labels,
            "missing": n_cells - ratings.n_labels,
            "categories": ratings.categories,
        },
        "settings": {"level": level, "weights": weights or "none", "threshold": threshold},
    }

    coeffs: dict = {}
    coeffs["percent_agreement"], _ = _safe(metrics.percent_agreement, ratings)
    fleiss_items, _, _ = metrics.fleiss_table(ratings)
    value, note = _safe(metrics.fleiss_kappa, ratings)
    coeffs["fleiss_kappa"] = value
    coeffs["fleiss_items"] = len(fleiss_items)
    coeffs["fleiss_raters"] = max(len(r) for r in ratings.by_item.values())
    if note:
        coeffs["fleiss_note"] = note
    value, note = _safe(metrics.krippendorff_alpha, ratings, level)
    coeffs["krippendorff_alpha"] = value
    if note:
        coeffs["krippendorff_note"] = note

    pairs = []
    for (a, b), (n, k) in metrics.pairwise_cohen(ratings, weights=weights).items():
        pairs.append({"a": a, "b": b, "shared": n, "kappa": None if math.isnan(k) else k})
    valid = [p["kappa"] for p in pairs if p["kappa"] is not None]
    coeffs["mean_pairwise_cohen_kappa"] = sum(valid) / len(valid) if valid else None
    report["coefficients"] = coeffs
    report["pairwise_cohen_kappa"] = pairs

    report["annotators"] = [
        s.to_dict() for s in annotator_vs_majority(ratings, threshold=threshold, min_items=min_items)
    ]
    contested = disagreement_report(ratings)
    report["disagreements"] = {
        "n_contested": len(contested),
        "items": [d.to_dict() for d in contested[:top]] if top else [d.to_dict() for d in contested],
    }

    if pair is None and len(annotators) == 2:
        pair = (annotators[0], annotators[1])
    if pair is not None:
        report["confusion"] = confusion_report(ratings, pair[0], pair[1], weights)
    return report


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _heading(title: str) -> list[str]:
    return ["", title, "-" * len(title)]


def format_report(report: dict, source: str | None = None) -> str:
    s = report["summary"]
    c = report["coefficients"]
    level = report["settings"]["level"]
    weights = report["settings"]["weights"]
    lines = ["Inter-annotator agreement report", "=" * 32]
    if source:
        lines.append(f"File:        {source}")
    lines.append(
        f"Items: {s['items']}   Annotators: {s['annotators']}   "
        f"Labels: {s['labels']} ({s['missing']} missing)"
    )
    lines.append(f"Categories:  {', '.join(s['categories'])}")

    lines += _heading("Agreement coefficients")
    kappa_name = "Mean pairwise Cohen's kappa" + ("" if weights == "none" else f" ({weights})")
    rows = [
        ("Percent agreement (pairwise)", c["percent_agreement"], None),
        (
            f"Fleiss' kappa ({c['fleiss_items']} items x {c['fleiss_raters']} raters)",
            c["fleiss_kappa"],
            landis_koch(c["fleiss_kappa"]),
        ),
        (f"Krippendorff's alpha ({level})", c["krippendorff_alpha"], None),
        (kappa_name, c["mean_pairwise_cohen_kappa"], landis_koch(c["mean_pairwise_cohen_kappa"])),
    ]
    width = max(len(r[0]) for r in rows)
    for name, value, band in rows:
        line = f"{name:<{width}}  {_fmt(value)}"
        if band and value is not None:
            line += f"  {band}"
        lines.append(line)
    alpha = c["krippendorff_alpha"]
    if alpha is not None:
        verdict = "reliable" if alpha >= 0.8 else "tentative" if alpha >= 0.667 else "unreliable"
        lines.append(f"(Krippendorff: alpha >= 0.800 reliable, >= 0.667 tentative -> {verdict})")

    if report["pairwise_cohen_kappa"]:
        lines += _heading("Pairwise Cohen's kappa")
        pw = max(len(f"{p['a']} - {p['b']}") for p in report["pairwise_cohen_kappa"])
        lines.append(f"{'pair':<{pw}}  shared  kappa")
        for p in report["pairwise_cohen_kappa"]:
            lines.append(f"{p['a'] + ' - ' + p['b']:<{pw}}  {p['shared']:>6}  {_fmt(p['kappa'])}")

    lines += _heading(
        f"Annotator vs. leave-one-out majority (flag below {report['settings']['threshold']:.2f})"
    )
    aw = max([len("annotator")] + [len(a["annotator"]) for a in report["annotators"]])
    lines.append(f"{'annotator':<{aw}}  labelled  compared  agree   rate  status")
    for a in report["annotators"]:
        status = "RETRAIN" if a["flagged"] else "ok" if a["agreement"] is not None else "-"
        lines.append(
            f"{a['annotator']:<{aw}}  {a['n_labelled']:>8}  {a['n_compared']:>8}  "
            f"{a['n_agree']:>5}  {_fmt(a['agreement'])}  {status}"
        )

    d = report["disagreements"]
    shown = len(d["items"])
    lines += _heading(f"Most contested items ({shown} of {d['n_contested']} with any disagreement)")
    if shown:
        iw = max([len("item")] + [len(x["item_id"]) for x in d["items"]])
        lines.append(f"{'item':<{iw}}  n  majority      share  entropy  labels")
        for x in d["items"]:
            maj = x["majority_label"] if x["majority_label"] is not None else "(tie)"
            dist = ", ".join(f"{k}:{v}" for k, v in x["label_counts"].items())
            lines.append(
                f"{x['item_id']:<{iw}}  {x['n_labels']}  {maj:<12}  {x['majority_share']:.2f}"
                f"  {x['entropy']:7.3f}  {dist}"
            )
    else:
        lines.append("All items are unanimous.")

    if "confusion" in report:
        lines.append("")
        lines.append(format_confusion(report["confusion"]))
    return "\n".join(lines)


def format_confusion(cm: dict) -> str:
    """Render a confusion-matrix dict (as produced by :func:`build_report`)."""
    lines = _heading(
        f"Confusion matrix: {cm['a']} (rows) vs {cm['b']} (columns), {cm['shared']} shared items"
    )[1:]
    cats = cm["categories"]
    cw = max(len(x) for x in cats)
    colw = max(5, *(len(x) for x in cats))
    lines.append(" " * cw + "  " + "  ".join(f"{x:>{colw}}" for x in cats))
    for cat, row in zip(cats, cm["matrix"]):
        lines.append(f"{cat:<{cw}}  " + "  ".join(f"{v:>{colw}}" for v in row))
    if "percent_agreement" in cm:
        lines.append("")
        lines.append(f"Percent agreement: {_fmt(cm['percent_agreement'])}")
        lines.append(f"Cohen's kappa:     {_fmt(cm['cohen_kappa'])}  {landis_koch(cm['cohen_kappa'])}")
    return "\n".join(lines)


def confusion_report(ratings: Ratings, a: str, b: str, weights: str | None = None) -> dict:
    """Confusion matrix, percent agreement and Cohen's kappa for one annotator pair."""
    known = ratings.annotators
    for name in (a, b):
        if name not in known:
            raise ValueError(f"unknown annotator {name!r}; known: {', '.join(known)}")
    items, la, lb = ratings.paired(a, b)
    if not items:
        raise ValueError(f"annotators {a!r} and {b!r} share no items")
    cats, matrix = metrics.confusion_matrix(la, lb, ratings.categories)
    kappa = metrics.cohen_kappa(la, lb, weights=weights, categories=cats)
    return {
        "a": a,
        "b": b,
        "shared": len(items),
        "categories": cats,
        "matrix": matrix,
        "percent_agreement": sum(x == y for x, y in zip(la, lb)) / len(la),
        "cohen_kappa": None if math.isnan(kappa) else kappa,
        "weights": weights or "none",
    }
