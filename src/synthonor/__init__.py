__version__ = "0.2.0"

from synthonor.fingerprints import FingerprintSpec
from synthonor.resources import example_space_path
from synthonor.product_search import ProductHit, ProductSearchEngine
from synthonor.synthon_or_rdkit import (
    ApproxHit,
    FingerprintCacheInfo,
    SearchConfig,
    SynthonORIndex,
    build_synthon_fingerprint_cache,
    default_fingerprint_cache_prefix,
    load_synthon_or_index,
    query_fingerprint_from_smiles,
    search_fingerprint,
    search_smiles,
    self_test_query_from_first_tuple,
    write_hits_jsonl,
)

__all__ = [
    "ApproxHit",
    "ProductHit",
    "ProductSearchEngine",
    "FingerprintSpec",
    "FingerprintCacheInfo",
    "SearchConfig",
    "SynthonORIndex",
    "__version__",
    "build_synthon_fingerprint_cache",
    "default_fingerprint_cache_prefix",
    "example_space_path",
    "load_synthon_or_index",
    "query_fingerprint_from_smiles",
    "search_fingerprint",
    "search_smiles",
    "self_test_query_from_first_tuple",
    "write_hits_jsonl",
]
