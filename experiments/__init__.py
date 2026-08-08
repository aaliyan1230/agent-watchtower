"""experiments: the seeded experiment suite and its analysis.

The paper's numbers come from here: detection rates per verifier,
false-positive rates on clean runs, judge agreement (Cohen's kappa).
Everything is a pure function of the artifacts — no hidden randomness.
"""

from .analyze import cohen_kappa, detection_rate, false_positive_rate
from .suite import ExperimentCell, build_grid

__all__ = ["cohen_kappa", "detection_rate", "false_positive_rate", "ExperimentCell", "build_grid"]
