# %% [markdown]
# # 001. Minimal Implementation
#
# This notebook rebuilds the core SynthonOR idea step by step on a tiny toy
# synthon table.
#
# Prerequisite: install `synthonor` first (`pip install synthonor` or `pip install -e .`).
#
# We will:
# 1. Load a handful of synthons from a TSV-like table.
# 2. Fingerprint every synthon.
# 3. Rank synthons inside each reaction slot against a query.
# 4. Enumerate tuples, combine fingerprints with OR-union, and rank results.
# 5. Compare our hand-written pipeline to `synthonor.search_smiles(...)`.
#
# The chemistry here is intentionally simple. The goal is to understand the
# mechanics, not to model a realistic reaction database.

# %%
from __future__ import annotations

import csv
from collections import defaultdict
from io import StringIO
from itertools import product
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

from synthonor import FingerprintSpec, build_synthon_fingerprint_cache, load_synthon_or_index, search_smiles
from synthonor.fingerprints import (
    fingerprint_smiles,
    tanimoto_similarity,
    union_fingerprints,
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
        subImgSize=(250, 180),
    )
    display(image)
    return image


fingerprint_spec = FingerprintSpec(kind="ecfp4").normalized()

# %% [markdown]
# ## 1. Create a tiny synthon table
#
# SynthonOR expects a tabular file with four ideas:
#
# - a synthon SMILES
# - a stable synthon identifier
# - a slot number inside a reaction
# - a reaction identifier
#
# For the hello-world version we will use one simple two-component reaction.
# Every synthon contains a `[U]` connector atom that marks the attachment point.

# %%
toy_space_tsv = """smiles\tsynthon_id\tposition\treaction_id
C[U]\ts1_methyl\t1\tr1
N[U]\ts1_amine\t1\tr1
O[U]\ts1_alcohol\t1\tr1
c1ccccc1[U]\ts1_phenyl\t1\tr1
C[U]\ts2_methyl\t2\tr1
CC[U]\ts2_ethyl\t2\tr1
C=C[U]\ts2_vinyl\t2\tr1
c1ccccc1[U]\ts2_phenyl\t2\tr1
"""

raw_rows = list(csv.DictReader(StringIO(toy_space_tsv), delimiter="\t"))
show_table(raw_rows)

# %% [markdown]
# ## 2. Fingerprint each synthon
#
# SynthonOR fingerprints every building block first. We will do the same by hand.

# %%
fingerprinted_rows: list[dict[str, object]] = []
for row in raw_rows:
    smiles = str(row["smiles"])
    fingerprinted_rows.append(
        {
            "reaction_id": str(row["reaction_id"]),
            "slot": int(row["position"]),
            "synthon_id": str(row["synthon_id"]),
            "smiles": smiles,
            "fp": fingerprint_smiles(smiles, fingerprint_spec),
        }
    )

show_table(
    [
        {
            "reaction_id": row["reaction_id"],
            "slot": row["slot"],
            "synthon_id": row["synthon_id"],
            "smiles": row["smiles"],
            "num_on_bits": row["fp"].GetNumOnBits(),
        }
        for row in fingerprinted_rows
    ]
)

# %% [markdown]
# ## 3. Group synthons by reaction and slot
#
# The real implementation builds an index so that each reaction is a tuple of
# slots, and each slot holds a bank of candidate synthons.

# %%
by_reaction_slot: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
for row in fingerprinted_rows:
    key = (str(row["reaction_id"]), int(row["slot"]))
    by_reaction_slot[key].append(row)

for key, rows in sorted(by_reaction_slot.items()):
    print(key, [row["synthon_id"] for row in rows])

# %% [markdown]
# ## 4. Fingerprint the query molecule
#
# We will search for `CCN`. In this toy space that should come from joining the
# `N[U]` synthon with the `CC[U]` synthon.

# %%
query_smiles = "CCN"
query_fp = fingerprint_smiles(query_smiles, fingerprint_spec)
print("Query:", query_smiles)
show_molecules([query_smiles], ["query"])

# %% [markdown]
# ## 5. Score synthons inside each slot
#
# SynthonOR first scores every synthon row against the query once, then each slot
# keeps only the synthons that look most similar under that shared score array.

# %%
slot_rankings: dict[tuple[str, int], list[dict[str, object]]] = {}
for key, rows in sorted(by_reaction_slot.items()):
    ranked = []
    for row in rows:
        ranked.append(
            {
                **row,
                "slot_score": tanimoto_similarity(query_fp, row["fp"], fingerprint_spec),
            }
        )
    ranked.sort(key=lambda item: (-float(item["slot_score"]), str(item["synthon_id"])))
    slot_rankings[key] = ranked

show_table(
    [
        {
            "reaction_id": reaction_id,
            "slot": slot,
            "synthon_id": row["synthon_id"],
            "smiles": row["smiles"],
            "slot_score": round(float(row["slot_score"]), 3),
        }
        for (reaction_id, slot), rows in slot_rankings.items()
        for row in rows
    ]
)

# %% [markdown]
# ## 6. Build candidate tuples and OR their fingerprints together
#
# For each possible tuple we combine the slot fingerprints with
# `union_fingerprints(...)`. For bit fingerprints such as ECFP4 this is just a
# bitwise OR.

# %%
reaction_id = "r1"
slot_1 = slot_rankings[(reaction_id, 1)][:3]
slot_2 = slot_rankings[(reaction_id, 2)][:3]

manual_candidates: list[dict[str, object]] = []
for synthon_a, synthon_b in product(slot_1, slot_2):
    combined_fp = union_fingerprints([synthon_a["fp"], synthon_b["fp"]], fingerprint_spec)
    approx_score = tanimoto_similarity(query_fp, combined_fp, fingerprint_spec)
    manual_candidates.append(
        {
            "reaction_id": reaction_id,
            "synthon_ids": (synthon_a["synthon_id"], synthon_b["synthon_id"]),
            "smiles": (synthon_a["smiles"], synthon_b["smiles"]),
            "approx_score": float(approx_score),
        }
    )

manual_candidates.sort(
    key=lambda item: (-float(item["approx_score"]), item["reaction_id"], item["synthon_ids"])
)

show_table(
    [
        {
            "rank": rank,
            "approx_score": round(candidate["approx_score"], 3),
            "synthon_ids": " + ".join(candidate["synthon_ids"]),
            "smiles": " + ".join(candidate["smiles"]),
        }
        for rank, candidate in enumerate(manual_candidates, start=1)
    ]
)

# %% [markdown]
# %% [markdown]
# ## 7. Compare with the packaged SynthonOR API
#
# The real library does the same ideas more cleanly and efficiently. We can write
# the same toy table to disk, build its packed fingerprint cache once, and confirm
# that `search_smiles(...)` gives the same winner.

# %%
generated_dir = ROOT / "notebooks" / "_generated"
generated_dir.mkdir(parents=True, exist_ok=True)
toy_space_path = generated_dir / "minimal_toy_space.tsv"
toy_space_path.write_text(toy_space_tsv, encoding="utf-8")

cache_info = build_synthon_fingerprint_cache(toy_space_path, fingerprint_spec=fingerprint_spec)
index = load_synthon_or_index(toy_space_path, fingerprint_spec=fingerprint_spec)
hits = search_smiles(
    query_smiles,
    index,
    top_n=6,
    preset="fast",
)

print("Toy cache prefix:", cache_info.cache_prefix)

show_table(
    [
        {
            "rank": hit.rank,
            "approx_score": round(hit.approx_score, 3),
            "reaction_id": hit.reaction_id,
            "synthon_ids": " + ".join(hit.synthon_ids),
            "smiles": " + ".join(hit.smiles),
        }
        for hit in hits
    ]
)

print("Manual top tuple:", manual_candidates[0]["synthon_ids"])
print("Library top tuple:", hits[0].synthon_ids)
assert tuple(manual_candidates[0]["synthon_ids"]) == hits[0].synthon_ids

# %% [markdown]
# Product assembly and exact rescoring are intentionally out of scope for this
# minimal tutorial notebook.
