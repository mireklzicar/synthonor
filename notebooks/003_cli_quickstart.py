# %% [markdown]
# # 003. CLI Quickstart
#
# A minimal command-line walkthrough for SynthonOR.
# Prerequisite: install `synthonor` first (`pip install synthonor` or `pip install -e .`).

# %%
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from synthonor import example_space_path

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT / "notebooks" / "_generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def resolve_example_space_path() -> Path:
    local_copy = GENERATED_DIR / "bundled_synthon_space_1M.tsv"
    if local_copy.exists():
        return local_copy
    with example_space_path() as bundled_path:
        shutil.copyfile(bundled_path, local_copy)
    return local_copy


DATA_PATH = resolve_example_space_path()
OUTPUT_PATH = GENERATED_DIR / "cli_quickstart_hits.jsonl"

# %% [markdown]
# ## 1. Build (or validate) the packed cache

# %%
cache_cmd = [
    "synthonor",
    str(DATA_PATH),
    "--fingerprint",
    "ecfp4",
    "--build-cache-only",
]
cache_proc = subprocess.run(cache_cmd, check=True, capture_output=True, text=True)
cache_summary = json.loads(cache_proc.stdout)
cache_summary

# %% [markdown]
# ## 2. Run a top-N search

# %%
search_cmd = [
    "synthonor",
    str(DATA_PATH),
    "--query",
    "CCOc1ccc(NC(=O)N2CCN(CC2)C)cc1",
    "--preset",
    "accurate",
    "--top-n",
    "5",
    "--output",
    str(OUTPUT_PATH),
]
search_proc = subprocess.run(search_cmd, check=True, capture_output=True, text=True)
search_summary = json.loads(search_proc.stdout)
search_summary

# %% [markdown]
# ## 3. Inspect output

# %%
OUTPUT_PATH.read_text(encoding="utf-8").splitlines()[:2]
