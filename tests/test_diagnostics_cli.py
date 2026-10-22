import json
from pathlib import Path

import pytest

from iaa import DataError, Ratings, annotator_vs_majority, disagreement_report, load_csv
from iaa.cli import main
from iaa.report import build_report, landis_koch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def make_ratings():
    # ann4 disagrees with the others on most items
    rows = []
    for i in range(10):
        truth = "pos" if i % 2 else "neg"
        for ann in ("ann1", "ann2", "ann3"):
            rows.append((f"it{i}", ann, truth))
        rows.append((f"it{i}", "ann4", truth if i < 3 else ("neg" if truth == "pos" else "pos")))
    return Ratings.from_records(rows)


def test_annotator_vs_majority_flags_outlier():
    scores = {s.annotator: s for s in annotator_vs_majority(make_ratings(), threshold=0.7)}
    assert scores["ann4"].agreement == pytest.approx(0.3)
    assert scores["ann4"].flagged
    assert not scores["ann1"].flagged
    assert scores["ann1"].agreement == pytest.approx(1.0)


def test_annotator_vs_majority_sorted_worst_first():
    scores = annotator_vs_majority(make_ratings())
    assert scores[0].annotator == "ann4"


def test_leave_one_out_skips_ties():
    r = Ratings.from_records([("1", "x", "a"), ("1", "y", "b"), ("1", "z", "c")])
    for s in annotator_vs_majority(r):
        assert s.n_compared == 0 and s.agreement is None and not s.flagged


def test_min_items_prevents_flagging_on_tiny_samples():
    r = Ratings.from_records([("1", "x", "a"), ("1", "y", "a"), ("1", "z", "b")])
    z = next(s for s in annotator_vs_majority(r, min_items=5) if s.annotator == "z")
    assert z.agreement == 0.0 and not z.flagged


def test_disagreement_report_orders_most_contested_first():
    r = Ratings.from_records(
        [("easy", "x", "a"), ("easy", "y", "a"), ("easy", "z", "a"),
         ("mid", "x", "a"), ("mid", "y", "a"), ("mid", "z", "b"),
         ("hard", "x", "a"), ("hard", "y", "b"), ("hard", "z", "c")]
    )
    rows = disagreement_report(r)
    assert [d.item_id for d in rows] == ["hard", "mid"]
    assert rows[0].majority_label is None  # three-way tie
    assert rows[1].majority_label == "a"
    assert rows[1].majority_share == pytest.approx(2 / 3)
    assert len(disagreement_report(r, include_unanimous=True)) == 3


def test_load_csv_handles_missing_and_validates(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("item_id,annotator,label\n1,a,x\n1,b,\n2,a,y\n", encoding="utf-8")
    r = load_csv(p)
    assert r.n_labels == 2
    assert r.items == ["1", "2"]

    bad = tmp_path / "bad.csv"
    bad.write_text("item,annotator,label\n1,a,x\n", encoding="utf-8")
    with pytest.raises(DataError, match="item_id"):
        load_csv(bad)


def test_conflicting_duplicate_label_rejected(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("item_id,annotator,label\n1,a,x\n1,a,y\n", encoding="utf-8")
    with pytest.raises(DataError, match="conflicting"):
        load_csv(p)


def test_landis_koch_bands():
    assert landis_koch(-0.1) == "poor"
    assert landis_koch(0.5) == "moderate"
    assert landis_koch(0.85) == "almost perfect"


def test_build_report_on_reference_data():
    report = build_report(load_csv(EXAMPLES / "krippendorff_reference.csv"))
    assert report["coefficients"]["krippendorff_alpha"] == pytest.approx(0.743, abs=5e-4)
    assert report["summary"]["missing"] == 7


def test_build_report_includes_confusion_for_two_annotators():
    r = Ratings.from_records([("1", "a", "x"), ("1", "b", "x"), ("2", "a", "y"), ("2", "b", "x")])
    report = build_report(r)
    assert report["confusion"]["matrix"] == [[1, 0], [1, 0]]


def test_cli_report_text(capsys):
    assert main(["report", str(EXAMPLES / "sentiment_labels.csv")]) == 0
    out = capsys.readouterr().out
    assert "Krippendorff's alpha (nominal)" in out
    assert "RETRAIN" in out


def test_cli_report_json_interval(capsys):
    assert main(["report", str(EXAMPLES / "krippendorff_reference.csv"), "--level", "interval", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["coefficients"]["krippendorff_alpha"] == pytest.approx(0.849, abs=5e-4)


def test_cli_confusion(capsys):
    assert main(["confusion", str(EXAMPLES / "sentiment_labels.csv"), "ann_a", "ann_d"]) == 0
    out = capsys.readouterr().out
    assert "Confusion matrix: ann_a (rows) vs ann_d (columns)" in out
    assert "Cohen's kappa" in out


def test_cli_errors_are_reported(tmp_path, capsys):
    assert main(["report", str(tmp_path / "nope.csv")]) == 2
    assert "iaa: error" in capsys.readouterr().err
    assert main(["confusion", str(EXAMPLES / "sentiment_labels.csv"), "ann_a", "ghost"]) == 2
