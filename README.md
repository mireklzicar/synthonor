# synthonor

Minimal standalone RDKit synthon-OR search with a packed on-disk synthon fingerprint index.

`synthonor` supports a simple workflow:

- build packed synthon fingerprints once per TSV + fingerprint setting
- memory-map the packed cache when searching
- reuse a valid cache automatically on later runs
- search with `load_synthon_or_index(...)`, `search_smiles(...)`, and `search_fingerprint(...)`

## Install

```bash
pip install synthonor
```

## Database Availability

SynthonOR expects a tab-separated table with these concepts:

```text
smiles\tsynthon_id\tposition\treaction_id
NC(=O)[C@@H]1CCCN1[U]\t100000003125\t1\t11a
C[C@@H](O)[C@H](N[U])C(N)=O\t100000003557\t1\t11a
CCCN([U])C(C)C(=O)Nc1ccccc1C\t100000003669\t1\t11a
O=C1CN([U])[C@@H](c2ccccc2)CO1\t100000005368\t1\t11a
```

The package ships with a small bundled example TSV (`synthon_space_1M.tsv`) as package data.

## Quick Start (Python)

```python
from synthonor import (
    FingerprintSpec,
    build_synthon_fingerprint_cache,
    example_space_path,
    load_synthon_or_index,
    search_smiles,
)

with example_space_path() as data_path:
    spec = FingerprintSpec(kind="ecfp4")
    cache_info = build_synthon_fingerprint_cache(data_path, fingerprint_spec=spec)
    index = load_synthon_or_index(data_path, fingerprint_spec=spec)

    hits = search_smiles(
        "CCOc1ccc(NC(=O)N2CCN(CC2)C)cc1",
        index,
        top_n=25,
        preset="accurate",
    )

print(cache_info.cache_prefix)
print(hits[0].reaction_id, hits[0].synthon_ids, round(hits[0].approx_score, 3))
```

Fingerprint-based search:

```python
from synthonor import query_fingerprint_from_smiles, search_fingerprint

query_fp = query_fingerprint_from_smiles("CCN1CCN(CC1)C(=O)c1ccccc1", index.fingerprint_spec)
hits = search_fingerprint(query_fp, index, min_score=0.35, preset="very_accurate")
```

## CLI

Build or validate cache only:

```bash
synthonor path/to/syntons.tsv \
  --fingerprint ecfp4 \
  --build-cache-only
```

Run search:

```bash
synthonor path/to/syntons.tsv \
  --query "CCOc1ccc(NC(=O)N2CCN(CC2)C)cc1" \
  --preset accurate \
  --top-n 25 \
  --output synthonor_hits.jsonl
```

Run explicit self-test mode:

```bash
synthonor path/to/syntons.tsv --test --preset fast --top-n 5
```

## Search Contract

- `top_n=N`: return at most `N` hits, sorted by descending approximate score.
- `min_score=S`: return every hit with approximate score `>= S`.
- `min_score=S, top_n=N`: apply score cutoff first, then cap to `N`.
- `max_score=T`: optionally bound score from above.
- returned `rank` values are ranks within the filtered output.

Config precedence:

- use `preset="fast" | "accurate" | "very_accurate"` for standard workflows
- pass `config=SearchConfig(...)` for explicit control
- explicit `config` overrides preset defaults
- explicit `top_n` overrides `config.topk_products`

## Search Presets

- `fast`: one reaction route, small candidate bank
- `accurate`: default setting
- `very_accurate`: searches all prescreened reactions

## Fingerprints

Packed on-disk synthon caches are used for bit fingerprint families:

- `ecfp4`
- `ecfp6`
- `rdkit`
- `patternfp`
- `atom_pair`
- `topological_torsion`

## Package Contents

After `pip install synthonor`, installed artifacts include:

- Python package code under `synthonor`
- bundled example TSV exposed via `synthonor.example_space_path()`

Repo-only artifacts that are not installed by default:

- local cache files you generate such as `*.synthon_fp_cache.*`
- notebooks in `notebooks/`
- local test/result outputs

## Tiny Benchmark (Bundled Example)

Measured on 2026-04-21 (Python 3.11, default `accurate` preset, query top-25) using the bundled example space (`6273` synthons, `42` reactions):

| Step | Time |
| --- | --- |
| first `load_synthon_or_index(...)` (cache build + mmap) | 0.57 s |
| repeated `load_synthon_or_index(...)` (cache reuse) | 0.02 s |
| `search_smiles(..., top_n=25, preset="accurate")` | 5.34 s |

Treat these as reference-scale numbers for this bundled dataset.

## Reproducible Scripts

From the repo root you can materialize the bundled synthon caches for every
bit-fingerprint family:

```bash
python synthonor/scripts/build_example_fingerprint_caches.py
```

And you can run a small exact full-product retrieval benchmark on the bundled
1M synthon slice, using the matching Freedom reaction table that already lives
in this repo:

```bash
python synthonor/scripts/run_exact_full_product_retrieval.py
```

Outputs land under `synthonor/results/` by default.

## Layout

- `src/synthonor/fingerprints.py`: fingerprint and similarity helpers
- `src/synthonor/synthon_or_rdkit.py`: cache build/load, index loading, search implementation
- `src/synthonor/resources.py`: bundled data helpers (`example_space_path`)
- `notebooks/001_minimal_implementation.py`: minimal end-to-end implementation
- `notebooks/002_basic_usage.py`: bundled database workflow
- `notebooks/003_cli_quickstart.py`: command-line quickstart
- `notebooks/004_adding_databases.py`: preparing custom TSV databases
- `notebooks/005_different_fingerprints.py`: comparing bit fingerprint families
