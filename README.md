# annotation-agreement

**Inter-annotator agreement for data-labelling QA: kappa, Fleiss, Krippendorff's alpha, plus the diagnostics you need to act on them.**

[![CI](https://github.com/ansariadeel-hue/annotation-agreement/actions/workflows/ci.yml/badge.svg)](https://github.com/ansariadeel-hue/annotation-agreement/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

## Why

In annotation and LLM-evaluation projects the label set is only as good as the
people producing it. Before a batch is accepted you want to know three things:

1. **Is the guideline working?** A single agreement number (kappa / alpha) for the batch.
2. **Which items are ambiguous?** Those go back for adjudication or a guideline update.
3. **Which annotator is drifting?** Someone consistently out of step with the
   rest of the team needs calibration or retraining, not silent averaging.

Spreadsheets usually only answer the first question, and often with plain
percent agreement, which ignores chance. `iaa` answers all three from a plain
`item_id, annotator, label` CSV export, handles missing labels, and has no
runtime dependencies beyond the Python standard library.

## Features

- **Percent agreement** (mean pairwise, per item)
- **Cohen's kappa** for two annotators, unweighted or with **linear / quadratic weights** for ordinal scales
- **Pairwise Cohen's kappa** for every annotator pair (on shared items) and their mean
- **Fleiss' kappa** for many annotators (uses items rated by the full panel)
- **Krippendorff's alpha** at the **nominal, ordinal, interval and ratio** levels, with **missing labels** handled natively
- **Per-item disagreement report**: majority label (ties detected), majority share and label entropy, most contested first
- **Per-annotator agreement vs. leave-one-out majority**, flagging annotators below a threshold for retraining
- **Confusion matrix** between any two annotators
- Landis & Koch verbal bands and Krippendorff's 0.667 / 0.800 reliability cut-offs in the report
- CLI with text or `--json` output; pure standard library

Correctness is checked against published reference values (see [Tests](#tests)).

## Install

```bash
pip install git+https://github.com/ansariadeel-hue/annotation-agreement
```

Requires Python 3.10+. No third-party runtime dependencies.

## Quickstart

```bash
iaa report examples/sentiment_labels.csv
```

Real output on the bundled example, a synthetic sentiment-labelling batch (24 items, 4 annotators, 3 classes, 6 missing labels):

```text
Inter-annotator agreement report
================================
File:        examples/sentiment_labels.csv
Items: 24   Annotators: 4   Labels: 90 (6 missing)
Categories:  negative, neutral, positive

Agreement coefficients
----------------------
Percent agreement (pairwise)         0.618
Fleiss' kappa (19 items x 4 raters)  0.442  moderate
Krippendorff's alpha (nominal)       0.420
Mean pairwise Cohen's kappa          0.416  moderate
(Krippendorff: alpha >= 0.800 reliable, >= 0.667 tentative -> unreliable)

Pairwise Cohen's kappa
----------------------
pair           shared  kappa
ann_a - ann_b      23  0.531
ann_a - ann_c      22  0.588
ann_a - ann_d      21  0.344
ann_b - ann_c      21  0.510
ann_b - ann_d      20  0.200
ann_c - ann_d      20  0.323

Annotator vs. leave-one-out majority (flag below 0.70)
------------------------------------------------------
annotator  labelled  compared  agree   rate  status
ann_d            21        20     12  0.600  RETRAIN
ann_b            23        19     15  0.789  ok
ann_c            22        20     16  0.800  ok
ann_a            24        20     17  0.850  ok

Most contested items (10 of 15 with any disagreement)
-----------------------------------------------------
item   n  majority      share  entropy  labels
rev06  3  (tie)         0.33    1.585  negative:1, neutral:1, positive:1
rev07  4  neutral       0.50    1.500  neutral:2, negative:1, positive:1
rev15  4  negative      0.50    1.500  negative:2, neutral:1, positive:1
rev21  4  (tie)         0.50    1.000  neutral:2, positive:2
rev01  3  neutral       0.67    0.918  neutral:2, negative:1
rev16  3  positive      0.67    0.918  positive:2, negative:1
rev03  4  positive      0.75    0.811  positive:3, negative:1
rev08  4  positive      0.75    0.811  positive:3, neutral:1
rev10  4  negative      0.75    0.811  negative:3, neutral:1
rev12  4  positive      0.75    0.811  positive:3, negative:1
```

Reading it: the batch is below Krippendorff's 0.667 threshold, every pair
involving `ann_d` has the lowest kappa, and `ann_d` agrees with the other
three annotators' majority only 60% of the time, so they are flagged for
retraining. The contested items list is the adjudication queue.

Drill into one pair:

```bash
iaa confusion examples/sentiment_labels.csv ann_a ann_d
```

```text
Confusion matrix: ann_a (rows) vs ann_d (columns), 21 shared items
------------------------------------------------------------------
          negative   neutral  positive
negative         4         1         2
neutral          0         7         1
positive         3         2         1

Percent agreement: 0.571
Cohen's kappa:     0.344  fair
```

Ordinal or numeric scales (e.g. 1-5 quality ratings): pick the measurement level and kappa weights:

```bash
iaa report examples/krippendorff_reference.csv --level interval --weights quadratic
```

### Options

| option | meaning |
| --- | --- |
| `--level {nominal,ordinal,interval,ratio}` | metric for Krippendorff's alpha (default `nominal`) |
| `--weights {none,linear,quadratic}` | weighting for Cohen's kappa (default `none`) |
| `--threshold 0.7` | flag annotators whose agreement with the majority is below this |
| `--min-items 5` | don't flag annotators compared on fewer items than this |
| `--top 10` | how many contested items to list (`0` = all) |
| `--pair A B` | append the confusion matrix for annotators A and B |
| `--json` | machine-readable output for dashboards / CI gates |

### Python API

```python
from iaa import load_csv, krippendorff_alpha, cohen_kappa, annotator_vs_majority

ratings = load_csv("examples/krippendorff_reference.csv")
krippendorff_alpha(ratings, "nominal")      # 0.7434...
krippendorff_alpha(ratings, "interval")     # 0.8491...
cohen_kappa(["1", "2", "3"], ["1", "3", "3"], weights="quadratic")
```

## Data format

Long format CSV, one row per (item, annotator) label, UTF-8:

```csv
item_id,annotator,label
rev01,ann_a,neutral
rev01,ann_b,neutral
rev01,ann_d,negative
```

- Required columns: `item_id`, `annotator`, `label` (extra columns are ignored).
- Missing labels: leave the row out, or leave `label` empty.
- A repeated `(item_id, annotator)` with a *different* label is rejected as an error.
- Numeric labels are ordered numerically (so `10` sorts after `9`); this order is
  used for weighted kappa and ordinal/interval/ratio alpha.

## How the numbers are defined

- **Percent agreement**: for each item, the share of annotator pairs that agree; averaged over items with 2+ labels.
- **Cohen's kappa**: `1 - sum(w * observed) / sum(w * expected)` with disagreement weights `w`
  (0/1 unweighted, `|i-j|/(k-1)` linear, `((i-j)/(k-1))^2` quadratic).
- **Fleiss' kappa**: Fleiss (1971); requires the same number of raters per item, so only items
  rated by the full panel are used (the report shows how many).
- **Krippendorff's alpha**: coincidence-matrix formulation, `alpha = 1 - (n-1) * sum(o_ck * delta_ck) / sum(n_c * n_k * delta_ck)`;
  units with a single value are unpairable and ignored.
- **Annotator vs. majority**: for each item an annotator labelled, the majority is computed from the
  *other* annotators only; items where they tie are skipped.

## Tests

```bash
python -m pytest -q
```

38 tests, including published reference values:

| check | reference | expected |
| --- | --- | --- |
| Krippendorff's alpha, 4 observers x 12 units with missing data | Krippendorff (2011), *Computing Krippendorff's Alpha-Reliability* | nominal 0.743, ordinal 0.815, interval 0.849, ratio 0.797 |
| Krippendorff's alpha, 2 coders, binary | same paper | 0.095 |
| Fleiss' kappa, 10 subjects x 14 raters x 5 categories | worked example in Wikipedia's *Fleiss' kappa* article | 0.210 |
| Cohen's kappa, 50 items (20/5/10/15) | worked example in Wikipedia's *Cohen's kappa* article | 0.400 |
| Quadratic weighted kappa | identity `2*cov / (var_a + var_b + (mean_a - mean_b)^2)` | matches |

The canonical Krippendorff data set ships as `examples/krippendorff_reference.csv`.

## Project structure

```text
annotation-agreement/
├── src/iaa/
│   ├── data.py          # CSV loading, Ratings container, category ordering
│   ├── metrics.py       # percent agreement, Cohen, Fleiss, Krippendorff, confusion matrix
│   ├── diagnostics.py   # per-item disagreement, annotator vs. majority
│   ├── report.py        # report assembly and text rendering
│   └── cli.py           # `iaa report`, `iaa confusion`
├── tests/
├── examples/
│   ├── sentiment_labels.csv          # 24 items x 4 annotators, with missing labels
│   └── krippendorff_reference.csv    # Krippendorff (2011) reliability data
├── pyproject.toml
└── LICENSE
```

## License

[MIT](LICENSE) © 2026 Ansari Adeel Ahmad
