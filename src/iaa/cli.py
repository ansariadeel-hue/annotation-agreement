"""Command-line interface: ``iaa report data.csv`` and ``iaa confusion data.csv A B``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, metrics
from .data import DataError, load_csv
from .report import build_report, confusion_report, format_confusion, format_report


def _weights(value: str) -> str | None:
    return None if value == "none" else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iaa", description="Inter-annotator agreement toolkit for data-labelling QA."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="full agreement report for a long-format CSV")
    rep.add_argument("csv", help="CSV with columns item_id, annotator, label")
    rep.add_argument("--level", choices=metrics.LEVELS, default="nominal",
                     help="measurement level for Krippendorff's alpha (default: nominal)")
    rep.add_argument("--weights", choices=["none", "linear", "quadratic"], default="none",
                     help="weighting for Cohen's kappa (default: none)")
    rep.add_argument("--threshold", type=float, default=0.7,
                     help="flag annotators whose agreement with the majority is below this (default: 0.7)")
    rep.add_argument("--min-items", type=int, default=5,
                     help="minimum compared items before an annotator can be flagged (default: 5)")
    rep.add_argument("--top", type=int, default=10,
                     help="number of contested items to list; 0 lists all (default: 10)")
    rep.add_argument("--pair", nargs=2, metavar=("A", "B"),
                     help="also print the confusion matrix between annotators A and B")
    rep.add_argument("--json", action="store_true", help="emit JSON instead of text")

    conf = sub.add_parser("confusion", help="confusion matrix + Cohen's kappa for two annotators")
    conf.add_argument("csv")
    conf.add_argument("a")
    conf.add_argument("b")
    conf.add_argument("--weights", choices=["none", "linear", "quadratic"], default="none")
    conf.add_argument("--json", action="store_true")
    return parser


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = build_parser().parse_args(argv)
    try:
        ratings = load_csv(args.csv)
        if args.command == "report":
            report = build_report(
                ratings,
                level=args.level,
                weights=_weights(args.weights),
                threshold=args.threshold,
                min_items=args.min_items,
                top=args.top,
                pair=tuple(args.pair) if args.pair else None,
            )
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(format_report(report, source=args.csv))
            return 0

        cm = confusion_report(ratings, args.a, args.b, _weights(args.weights))
        print(json.dumps(cm, indent=2, ensure_ascii=False) if args.json else format_confusion(cm))
        return 0
    except (DataError, ValueError, OSError) as exc:
        print(f"iaa: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
