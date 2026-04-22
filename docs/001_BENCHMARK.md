# Exact Full-Product Retrieval Benchmark

This note summarizes the reproducible full-product retrieval benchmark for the
bundled SynthonOR example space:

- bundled synthon TSV:
  [synthon_space_1M.tsv](/home/mirekl/Synthon_RL/synthonor/src/synthonor/data/synthon_space_1M.tsv)
- reaction table:
  [2024-09_Freedom_reactions.txt](/home/mirekl/Synthon_RL/synthongpt_beta/data/molecules/chemspace/2024-09_Freedom_reactions.txt)
- exact benchmark outputs:
  [summary.json](/home/mirekl/Synthon_RL/synthonor/results/full_product_retrieval_fingerprint_kinds/summary.json)
  [exact_ground_truth_top100.csv](/home/mirekl/Synthon_RL/synthonor/results/full_product_retrieval_fingerprint_kinds/exact_ground_truth_top100.csv)
  [approx_retrievals.csv](/home/mirekl/Synthon_RL/synthonor/results/full_product_retrieval_fingerprint_kinds/approx_retrievals.csv)

## Setup

The benchmark mirrors the earlier `synthonor_beta/COMPARISON.md` style, but
uses the main `synthonor` implementation and compares multiple fingerprint
families.

Preset note:

- this benchmark uses the `accurate` SynthonOR search preset
- equivalently, it uses the same `SearchConfig` values as
  `SearchConfig.preset("accurate")`

- `42` deterministic queries
- one sampled real synthon tuple per bundled reaction
- exact truth defined by full-product ECFP4 / Morgan radius `2`
- search budget:
  - `reaction_cap = 8`
  - `candidate_bank_size = 192`
  - `exhaustive_tuple_limit = 250000`
  - `topk_products = 100`

Exact-library summary:

- total assembled tuples: `990,829`
- duplicate products collapsed away: `54,732`
- final exact full-product library: `936,097` unique molecules

## Repro Commands

From the repo root:

```bash
python synthonor/scripts/build_example_fingerprint_caches.py
python synthonor/scripts/run_exact_full_product_retrieval.py
python synthonor/scripts/plot_full_product_retrieval_distributions.py
```

## Headline Results

| fingerprint | mean exact top100 overlap | mean exact top1 similarity | mean wall time / query (s) |
| --- | ---: | ---: | ---: |
| `ecfp4` | `63.38` | `0.9729` | `4.03` |
| `ecfp6` | `56.38` | `0.9724` | `3.76` |
| `topological_torsion` | `45.07` | `0.9792` | `4.19` |
| `rdkit` | `29.26` | `0.8699` | `5.06` |
| `atom_pair` | `17.31` | `0.6405` | `5.38` |
| `patternfp` | `3.76` | `0.3476` | `5.25` |

## Size Statistics

The table below compares the exact ground-truth nearest neighbors against the
SynthonOR retrieved products after count matching per query, so each fingerprint
is compared on the same number of products per query.

| fingerprint | ground truth mean MW | synthonor mean MW | ground truth mean atom count | synthonor mean atom count |
| --- | ---: | ---: | ---: | ---: |
| `ecfp4` | `415.12` | `417.52` | `28.99` | `29.12` |
| `ecfp6` | `415.10` | `414.34` | `28.98` | `28.79` |
| `topological_torsion` | `415.14` | `431.09` | `28.97` | `30.34` |
| `rdkit` | `415.75` | `468.05` | `29.00` | `33.06` |
| `atom_pair` | `417.10` | `560.56` | `29.07` | `40.13` |
| `patternfp` | `416.85` | `573.05` | `29.04` | `40.99` |

## Short Takeaways

- `ecfp4` is still the strongest overall retrieval fingerprint on this bundled
  1M slice.
- `ecfp6` is the nearest alternative, but it gives up some deeper top100
  overlap.
- `topological_torsion` is interesting because its exact top1 quality is very
  strong, even though its deeper recall is worse than `ecfp4` and `ecfp6`.
- `rdkit`, `atom_pair`, and especially `patternfp` are clearly weaker for this
  exact full-product nearest-neighbor objective.

## Distribution Plots

The plotting script writes one count-matched histogram + KDE panel per
fingerprint under:

- [plots](/home/mirekl/Synthon_RL/synthonor/results/full_product_retrieval_fingerprint_kinds/plots)

Each plot overlays:

- `ground_truth`: the top exact nearest neighbors, truncated per query to match
  the number of unique SynthonOR products retrieved for that fingerprint
- `<fingerprint>`: the exact ECFP4 similarities of the SynthonOR retrieved
  products after full-product assembly and deduplication
