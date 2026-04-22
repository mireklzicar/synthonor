# %% [markdown]
# # 004. Adding Databases
#
# This notebook shows how to make new synthon tables compatible with SynthonOR.
# Prerequisite: install `synthonor` first (`pip install synthonor` or `pip install -e .`).
#
# We will:
#
# 1. Create a tiny custom database from scratch.
# 2. Use both canonical and alternate column names accepted by the loader.
# 3. Merge two databases into one search space.
# 4. Handle imperfect input with `skip_bad_rows=True`.

# %%
from __future__ import annotations

import csv
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

from synthonor import FingerprintSpec, build_synthon_fingerprint_cache, load_synthon_or_index, search_smiles


def show_table(rows: list[dict[str, object]]):
    if pd is not None:
        frame = pd.DataFrame(rows)
        display(frame)
        return frame
    for row in rows:
        pprint(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


OUTPUT_DIR = ROOT / "notebooks" / "_generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPEC = FingerprintSpec(kind="ecfp4").normalized()

# %% [markdown]
# ## 1. The minimum schema
#
# `load_synthon_or_index(...)` looks for four concepts:
#
# - SMILES: `SMILES` or `smiles`
# - synthon ID: `synthon_id`, `synton_id`, or `id`
# - slot: `synthon#`, `synton#`, `slot`, or `position`
# - reaction ID: `reaction_id`, `rxn_id`, or `reaction`
#
# Here is a clean custom database using the canonical lowercase names.

# %%
custom_rows_a = [
    {"smiles": "C[U]", "synthon_id": "a_methyl", "position": "1", "reaction_id": "r_custom_a"},
    {"smiles": "N[U]", "synthon_id": "a_amine", "position": "1", "reaction_id": "r_custom_a"},
    {"smiles": "CC[U]", "synthon_id": "b_ethyl", "position": "2", "reaction_id": "r_custom_a"},
    {"smiles": "C=C[U]", "synthon_id": "b_vinyl", "position": "2", "reaction_id": "r_custom_a"},
]
custom_path_a = write_rows(
    OUTPUT_DIR / "custom_space_a.tsv",
    custom_rows_a,
    fieldnames=["smiles", "synthon_id", "position", "reaction_id"],
)
show_table(custom_rows_a)

# %%
cache_a = build_synthon_fingerprint_cache(custom_path_a, fingerprint_spec=SPEC)
print("Custom cache A:", cache_a.cache_prefix)

# %%
index_a = load_synthon_or_index(custom_path_a, fingerprint_spec=SPEC)
hits_a = search_smiles(
    "CCN",
    index_a,
    top_n=5,
    preset="fast",
)
show_table(
    [
        {
            "rank": hit.rank,
            "approx_score": round(hit.approx_score, 3),
            "reaction_id": hit.reaction_id,
            "synthon_ids": " | ".join(hit.synthon_ids),
        }
        for hit in hits_a
    ]
)

# %% [markdown]
# ## 2. Alternate column names work too
#
# This second file uses different header names, but they still map to the same
# required concepts.

# %%
custom_rows_b = [
    {"SMILES": "O[U]", "id": "c_alcohol", "slot": "1", "reaction": "r_custom_b"},
    {"SMILES": "c1ccccc1[U]", "id": "c_phenyl", "slot": "1", "reaction": "r_custom_b"},
    {"SMILES": "C[U]", "id": "d_methyl", "slot": "2", "reaction": "r_custom_b"},
    {"SMILES": "CC[U]", "id": "d_ethyl", "slot": "2", "reaction": "r_custom_b"},
]
custom_path_b = write_rows(
    OUTPUT_DIR / "custom_space_b.tsv",
    custom_rows_b,
    fieldnames=["SMILES", "id", "slot", "reaction"],
)
show_table(custom_rows_b)

# %%
cache_b = build_synthon_fingerprint_cache(custom_path_b, fingerprint_spec=SPEC)
print("Custom cache B:", cache_b.cache_prefix)

# %%
index_b = load_synthon_or_index(custom_path_b, fingerprint_spec=SPEC)
print("Database B reactions:", index_b.num_reactions)
print("Database B synthons:", index_b.num_synthons)

# %% [markdown]
# ## 3. Merge databases into one search space
#
# A merged file is usually just concatenated rows written back out with a single,
# normalized header.

# %%
normalized_rows: list[dict[str, str]] = []
for row in custom_rows_a:
    normalized_rows.append(dict(row))
for row in custom_rows_b:
    normalized_rows.append(
        {
            "smiles": row["SMILES"],
            "synthon_id": row["id"],
            "position": row["slot"],
            "reaction_id": row["reaction"],
        }
    )

merged_path = write_rows(
    OUTPUT_DIR / "merged_custom_space.tsv",
    normalized_rows,
    fieldnames=["smiles", "synthon_id", "position", "reaction_id"],
)
merged_cache = build_synthon_fingerprint_cache(merged_path, fingerprint_spec=SPEC)
merged_index = load_synthon_or_index(merged_path, fingerprint_spec=SPEC)
merged_hits = search_smiles(
    "CCO",
    merged_index,
    top_n=8,
    preset="accurate",
)

show_table(
    [
        {
            "rank": hit.rank,
            "approx_score": round(hit.approx_score, 3),
            "reaction_id": hit.reaction_id,
            "synthon_ids": " | ".join(hit.synthon_ids),
        }
        for hit in merged_hits
    ]
)
print("Merged cache:", merged_cache.cache_prefix)

# %% [markdown]
# ## 4. Clean up imperfect rows
#
# By default the loader raises an error when a row is malformed. That is the
# safest behavior while you are validating a new database.

# %%
bad_rows = normalized_rows + [
    {"smiles": "not_a_smiles", "synthon_id": "broken", "position": "1", "reaction_id": "r_bad"},
    {"smiles": "C[U]", "synthon_id": "missing_slot", "position": "", "reaction_id": "r_bad"},
]
bad_path = write_rows(
    OUTPUT_DIR / "merged_with_bad_rows.tsv",
    bad_rows,
    fieldnames=["smiles", "synthon_id", "position", "reaction_id"],
)

try:
    load_synthon_or_index(bad_path, fingerprint_spec=SPEC)
except Exception as exc:
    print(type(exc).__name__, exc)

# %% [markdown]
# When you want a best-effort load instead, pass `skip_bad_rows=True`.

# %%
cleaned_cache = build_synthon_fingerprint_cache(bad_path, fingerprint_spec=SPEC, skip_bad_rows=True, overwrite=True)
cleaned_index = load_synthon_or_index(bad_path, fingerprint_spec=SPEC, skip_bad_rows=True)
print("Synthons after skipping bad rows:", cleaned_index.num_synthons)
print("Reactions after skipping bad rows:", cleaned_index.num_reactions)
print("Cleaned cache:", cleaned_cache.cache_prefix)

# %% [markdown]
# ## 5. Practical checklist
#
# Before adding a new production database, it is worth checking:
#
# 1. Every reaction has the slots you expect.
# 2. Connector atoms are encoded consistently.
# 3. Synthon IDs are stable enough for downstream reporting.
# 4. You built the packed fingerprint cache for the fingerprint you plan to search with.
# 5. A few smoke-test queries return sensible reactions and synthons.
