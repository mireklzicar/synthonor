from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SYNTHONOR_ROOT = SCRIPT_DIR.parent

for candidate in (SYNTHONOR_ROOT / "src",):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from synthonor import FingerprintSpec  # noqa: E402


DEFAULT_TSV_PATH = SYNTHONOR_ROOT / "src" / "synthonor" / "data" / "synthon_space_1M.tsv"
DEFAULT_REACTIONS_PATH = SYNTHONOR_ROOT / "src" / "synthonor" / "data" / "synthon_space_1M.reactions.tsv"
DEFAULT_CACHE_OUTPUT_DIR = SYNTHONOR_ROOT / "results" / "bundled_1M_fingerprint_caches"
DEFAULT_BENCHMARK_OUTPUT_DIR = SYNTHONOR_ROOT / "results" / "full_product_retrieval_fingerprint_kinds"

BIT_FINGERPRINT_SPECS = (
    FingerprintSpec(kind="ecfp4"),
    FingerprintSpec(kind="ecfp6"),
    FingerprintSpec(kind="rdkit"),
    FingerprintSpec(kind="patternfp"),
    FingerprintSpec(kind="atom_pair"),
    FingerprintSpec(kind="topological_torsion"),
)

COUNT_FINGERPRINT_SPECS = (
    FingerprintSpec(kind="morgan_counts", union_mode="sum"),
    FingerprintSpec(kind="morgan_counts", union_mode="max"),
)

BENCHMARK_FINGERPRINT_SPECS = BIT_FINGERPRINT_SPECS
ALL_FINGERPRINT_SPECS = BIT_FINGERPRINT_SPECS + COUNT_FINGERPRINT_SPECS
EXACT_FINGERPRINT_SPEC = FingerprintSpec(kind="ecfp4").normalized()


def fingerprint_labels(specs: tuple[FingerprintSpec, ...]) -> list[str]:
    return [spec.normalized().label for spec in specs]
