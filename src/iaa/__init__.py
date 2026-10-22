"""iaa - inter-annotator agreement toolkit for data-labelling QA."""

from .data import DataError, Ratings, load_csv
from .diagnostics import annotator_vs_majority, disagreement_report
from .metrics import (
    cohen_kappa,
    confusion_matrix,
    fleiss_kappa,
    fleiss_kappa_from_table,
    krippendorff_alpha,
    krippendorff_alpha_units,
    pairwise_cohen,
    percent_agreement,
)

__version__ = "0.1.0"

__all__ = [
    "DataError",
    "Ratings",
    "load_csv",
    "annotator_vs_majority",
    "disagreement_report",
    "cohen_kappa",
    "confusion_matrix",
    "fleiss_kappa",
    "fleiss_kappa_from_table",
    "krippendorff_alpha",
    "krippendorff_alpha_units",
    "pairwise_cohen",
    "percent_agreement",
    "__version__",
]
