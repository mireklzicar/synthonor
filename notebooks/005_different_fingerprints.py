# %% [markdown]
# # 005. Different Fingerprints
#
# SynthonOR lets you swap bit fingerprint families while keeping the connector
# placeholders intact. This notebook compares a few practical cache-backed
# choices on the bundled example database.
#
# Prerequisite: install `synthonor` first (`pip install synthonor` or `pip install -e .`).
#
# We will look at:
#
# - different bit fingerprint families
# - the packed cache created for each one
#
# To keep the notebook responsive, we first carve out a small reaction subset
# from the bundled TSV and run all comparisons on that slice.

# %%
from __future__ import annotations

import csv
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
    search_smiles,
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


def write_reaction_subset(source_path: Path, target_path: Path, *, reaction_limit: int = 8) -> Path:
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {source_path}")

        selected_reactions: list[str] = []
        selected_rows: list[dict[str, str]] = []
        for row in reader:
            reaction_id = str(row["reaction_id"])
            if reaction_id not in selected_reactions:
                if len(selected_reactions) >= reaction_limit:
                    continue
                selected_reactions.append(reaction_id)
            if reaction_id in selected_reactions:
                selected_rows.append({key: str(value) for key, value in row.items()})

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected_rows)

    print("Subset reactions:", selected_reactions)
    print("Subset rows:", len(selected_rows))
    return target_path


def resolve_example_space_path() -> Path:
    generated_dir = ROOT / "notebooks" / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    local_copy = generated_dir / "bundled_synthon_space_1M.tsv"
    if local_copy.exists():
        return local_copy
    with example_space_path() as bundled_path:
        shutil.copyfile(bundled_path, local_copy)
    return local_copy


FULL_DATA_PATH = resolve_example_space_path()
GENERATED_DIR = ROOT / "notebooks" / "_generated"
COMPARE_DATA_PATH = write_reaction_subset(
    FULL_DATA_PATH,
    GENERATED_DIR / "fingerprint_comparison_subset.tsv",
    reaction_limit=8,
)
QUERY = "CCOc1ccc(NC(=O)N2CCN(CC2)C)cc1"
PRESET = "accurate"
TOP_N = 8

# %% [markdown]
# ## 1. Compare several cache-backed fingerprint families
#
# Every fingerprint setting below gets its own packed synthon cache on disk.

# %%
fingerprint_specs = [
    FingerprintSpec(kind="ecfp4"),
    FingerprintSpec(kind="ecfp6"),
    FingerprintSpec(kind="rdkit"),
    FingerprintSpec(kind="patternfp"),
    FingerprintSpec(kind="atom_pair"),
]

family_rows: list[dict[str, object]] = []
family_hits = {}
for spec in fingerprint_specs:
    active_spec = spec.normalized()
    cache_info = build_synthon_fingerprint_cache(COMPARE_DATA_PATH, fingerprint_spec=active_spec)
    index = load_synthon_or_index(COMPARE_DATA_PATH, fingerprint_spec=active_spec)
    hits = search_smiles(QUERY, index, top_n=TOP_N, preset=PRESET)
    family_hits[active_spec.label] = hits
    top = hits[0]
    family_rows.append(
        {
            "fingerprint": active_spec.label,
            "cache_prefix": Path(cache_info.cache_prefix).name,
            "top_score": round(top.approx_score, 3),
            "top_reaction": top.reaction_id,
            "top_synthon_ids": " | ".join(top.synthon_ids),
        }
    )

show_table(family_rows)

# %% [markdown]
# ## 2. Visualize the top hit under a few settings
#
# A quick way to build intuition is to compare which synthons each fingerprint
# family prefers for the same query.

# %%
selected_labels = ["ecfp4", "ecfp6", "atom_pair"]
for label in selected_labels:
    hit = family_hits[label][0]
    print(label, "->", hit.reaction_id, hit.synthon_ids, round(hit.approx_score, 3))
    show_molecules(
        [QUERY] + list(hit.smiles),
        [f"query ({label})"] + [f"slot {slot}" for slot in hit.slots],
        mols_per_row=2,
    )

# %% [markdown]
# ## 3. Takeaways
#
# A good starting order is:
#
# 1. `ecfp4` for a simple baseline.
# 2. `ecfp6` when you want a slightly wider circular neighborhood.
# 3. `patternfp`, `rdkit`, or `atom_pair` as quick comparison baselines.
