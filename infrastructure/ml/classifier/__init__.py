"""
Significant-Move Classifier
===========================

Supervised counterpart to the unsupervised models in
:mod:`infrastructure.ml`.  Predicts, at the close of day ``t``, whether
day ``t+1`` will deliver an abnormally large absolute move for a given
F&O symbol — where "abnormally large" is scaled by that symbol's own
trailing volatility.

Design points that matter more than the model choice
----------------------------------------------------

* **Vol-scaled target.**  An absolute threshold would just teach the
  model which symbols are volatile.  See :mod:`labeling`.
* **Date-based purged walk-forward CV.**  Row-based splitters cut a
  trading date in half and leak market-wide shocks sideways across the
  boundary.  See :mod:`cv`.
* **Tuning is confined to a development period.**  Optuna never sees the
  blocks the reported lift is computed on.  See :mod:`tuning`.
* **Lift is quoted against a fitted linear baseline**, not against
  nothing.  See :mod:`baseline`.

Public API
----------
    from infrastructure.ml.classifier import run
    result = run()
    print(result.summary)
"""

from infrastructure.ml.classifier.cv import (
    Fold,
    PurgedWalkForward,
    assert_no_leakage,
)
from infrastructure.ml.classifier.evaluate import Metrics, score
from infrastructure.ml.classifier.explain import ShapResult, explain
from infrastructure.ml.classifier.features import (
    build_feature_panel,
    feature_columns,
    load_universe,
)
from infrastructure.ml.classifier.labeling import add_labels, label_summary
from infrastructure.ml.classifier.models import FittedModel, train
from infrastructure.ml.classifier.pipeline import (
    RunResult,
    build_dataset,
    prepare,
    run,
    walk_forward,
)
from infrastructure.ml.classifier.tuning import TuningResult, tune

__all__ = [
    # orchestration
    "run",
    "RunResult",
    "build_dataset",
    "prepare",
    "walk_forward",
    # data
    "build_feature_panel",
    "feature_columns",
    "load_universe",
    "add_labels",
    "label_summary",
    # validation
    "PurgedWalkForward",
    "Fold",
    "assert_no_leakage",
    # modelling
    "train",
    "FittedModel",
    "tune",
    "TuningResult",
    # analysis
    "score",
    "Metrics",
    "explain",
    "ShapResult",
]
