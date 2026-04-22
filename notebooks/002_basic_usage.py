# %% [markdown]
# # 002. Basic Usage
#
# This notebook shows the beginner-friendly SynthonOR workflow on the bundled
# example synthon space.
#
# Prerequisite: install `synthonor` first (`pip install synthonor` or `pip install -e .`).
#
# The common patterns are:
#
# 1. `top_n=...`: give me the best N hits.
# 2. `min_score=...`: give me everything above a Tanimoto threshold.
# 3. both together: give me up to N hits above a threshold.

# %%
from __future__ import annotations

import shutil
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

from rdkit import Chem
from rdkit.Chem import Draw

from synthonor import (
    FingerprintSpec,
    build_synthon_fingerprint_cache,
    example_space_path,
    load_synthon_or_index,
    query_fingerprint_from_smiles,
    search_fingerprint,
    search_smiles,
    self_test_query_from_first_tuple,
    write_hits_jsonl,
)


def show_table(rows: list[dict[str, object]]):
    if pd is not None:
        frame = pd.DataFrame(rows)
        display(frame)
        return frame
    for row in rows:
        pprint(row)
    return rows


def show_molecules(smiles_list: list[str], legends: list[str], *, mols_per_row: int = 4):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    image = Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=mols_per_row,
        subImgSize=(260, 190),
    )
    display(image)
    return image


def resolve_example_space_path() -> Path:
    generated_dir = ROOT / "notebooks" / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    local_copy = generated_dir / "bundled_synthon_space_1M.tsv"
    if local_copy.exists():
        return local_copy
    with example_space_path() as bundled_path:
        shutil.copyfile(bundled_path, local_copy)
    return local_copy


DATA_PATH = resolve_example_space_path()
fingerprint_spec = FingerprintSpec(kind="ecfp4").normalized()
PRESET = "accurate"

# %% [markdown]
# ## 1. Build or reuse a local packed fingerprint cache
#
# The installed package ships the example TSV, not a prebuilt cache. Calling
# `build_synthon_fingerprint_cache(...)` here creates the local cache on first
# run and then reuses it on later runs if the TSV still matches.

# %%
cache_info = build_synthon_fingerprint_cache(DATA_PATH, fingerprint_spec=fingerprint_spec)
index = load_synthon_or_index(DATA_PATH, fingerprint_spec=fingerprint_spec)
print("Database:", DATA_PATH)
print("Cache prefix:", cache_info.cache_prefix)
print("Packed fingerprints:", cache_info.packed_fps_path)
print("Bit counts:", cache_info.bitcounts_path)
print("Reactions:", index.num_reactions)
print("Slots:", index.num_slots)
print("Synthons:", index.num_synthons)
print("Fingerprint:", index.fingerprint_spec.label)
print("Default preset for this notebook:", PRESET)

# %% [markdown]
# ## 2. Top-N search
#
# This is the simplest search mode: ask for the best `N` approximate hits.
#
# `preset="accurate"` is the library default, so you can omit it, but keeping it
# visible here makes the intent explicit.

# %%
query_smiles = "CCOc1ccc(NC(=O)N2CCN(CC2)C)cc1"

top_hits = search_smiles(
    query_smiles,
    index,
    top_n=10,
    preset=PRESET,
)

show_table(
    [
        {
            "rank": hit.rank,
            "approx_score": round(hit.approx_score, 3),
            "reaction_id": hit.reaction_id,
            "synthon_ids": " | ".join(hit.synthon_ids),
            "slots": " | ".join(str(slot) for slot in hit.slots),
        }
        for hit in top_hits
    ]
)

# %% [markdown]
# ## 3. Threshold search
#
# Sometimes you care more about a score cutoff than a fixed hit count.
#
# This returns every hit with approximate Tanimoto score at least `0.35`.

# %%
threshold_hits = search_smiles(
    query_smiles,
    index,
    min_score=0.35,
    preset=PRESET,
)

print("Hits with score >= 0.35:", len(threshold_hits))
show_table(
    [
        {
            "rank": hit.rank,
            "approx_score": round(hit.approx_score, 3),
            "reaction_id": hit.reaction_id,
            "synthon_ids": " | ".join(hit.synthon_ids),
        }
        for hit in threshold_hits[:10]
    ]
)

# %% [markdown]
# ## 4. Combined search: threshold + cap
#
# This is a common practical pattern: keep only hits above a minimum score, but
# do not return an unbounded list.

# %%
combined_hits = search_smiles(
    query_smiles,
    index,
    min_score=0.35,
    top_n=5,
    preset=PRESET,
)

show_table(
    [
        {
            "rank": hit.rank,
            "approx_score": round(hit.approx_score, 3),
            "reaction_id": hit.reaction_id,
            "synthon_ids": " | ".join(hit.synthon_ids),
        }
        for hit in combined_hits
    ]
)

# %% [markdown]
# The returned objects are `ApproxHit` records. Each one stores the reaction,
# ranked score, synthon IDs, slot numbers, synthon SMILES, and source row IDs.

# %%
top_hit = top_hits[0]
pprint(top_hit)

# %% [markdown]
# ## 5. Visualize the query and the top hit's synthons
#
# At this stage SynthonOR has ranked *tuples of synthons*. Without a reaction
# schema file we cannot assemble products here, but we can still inspect the
# query and the building blocks that explain the hit.

# %%
show_molecules(
    [query_smiles] + list(top_hit.smiles),
    ["query"] + [f"slot {slot}: {synthon_id}" for slot, synthon_id in zip(top_hit.slots, top_hit.synthon_ids)],
    mols_per_row=2,
)

# %% [markdown]
# ## 6. Search from a precomputed fingerprint
#
# If you already have a fingerprint in memory, use `search_fingerprint(...)`
# with the same `top_n` / `min_score` style arguments.

# %%
query_fp = query_fingerprint_from_smiles(query_smiles, index.fingerprint_spec)
fp_hits = search_fingerprint(query_fp, index, top_n=10, preset=PRESET)

print("Top tuple from search_smiles:", top_hits[0].synthon_ids)
print("Top tuple from search_fingerprint:", fp_hits[0].synthon_ids)
assert top_hits[0].synthon_ids == fp_hits[0].synthon_ids

# %% [markdown]
# ## 7. Use a built-in self-test query
#
# `self_test_query_from_first_tuple(...)` is handy when you want to smoke-test a
# database load and search pipeline without picking a real product up front.

# %%
self_test_fp = self_test_query_from_first_tuple(index)
self_test_hits = search_fingerprint(self_test_fp, index, top_n=5, preset="fast")
show_table(
    [
        {
            "rank": hit.rank,
            "approx_score": round(hit.approx_score, 3),
            "reaction_id": hit.reaction_id,
            "synthon_ids": " | ".join(hit.synthon_ids),
        }
        for hit in self_test_hits
    ]
)

# %% [markdown]
# ## 8. Save results
#
# The CLI writes JSONL, and the Python API can do the same.

# %%
generated_dir = ROOT / "notebooks" / "_generated"
generated_dir.mkdir(parents=True, exist_ok=True)
output_path = generated_dir / "basic_usage_hits.jsonl"
write_hits_jsonl(top_hits, output_path)
print("Wrote", output_path)
print(output_path.read_text(encoding="utf-8").splitlines()[0])

# %% [markdown]
# ## 9. Build a cache for a new TSV
#
# For a new database, build the packed synthon cache once and then load it with
# the same `load_synthon_or_index(...)` call you already used above.

# %%
generated_space = generated_dir / "basic_usage_toy_space.tsv"
generated_space.write_text(
    "\n".join(
        [
            "smiles\tsynthon_id\tposition\treaction_id",
            "C[U]\ta1\t1\tr_cache_demo",
            "N[U]\ta2\t1\tr_cache_demo",
            "CC[U]\tb1\t2\tr_cache_demo",
            "O[U]\tb2\t2\tr_cache_demo",
        ]
    )
    + "\n",
    encoding="utf-8",
)

toy_cache = build_synthon_fingerprint_cache(
    generated_space,
    fingerprint_spec=FingerprintSpec(kind="ecfp4"),
)
toy_index = load_synthon_or_index(generated_space, fingerprint_spec=FingerprintSpec(kind="ecfp4"))
toy_hits = search_smiles("CCN", toy_index, top_n=3, preset="accurate")

print("Toy cache prefix:", toy_cache.cache_prefix)
show_table(
    [
        {
            "rank": hit.rank,
            "approx_score": round(hit.approx_score, 3),
            "synthon_ids": " | ".join(hit.synthon_ids),
        }
        for hit in toy_hits
    ]
)
