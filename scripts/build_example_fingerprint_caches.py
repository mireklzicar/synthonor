#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _common import BIT_FINGERPRINT_SPECS, DEFAULT_CACHE_OUTPUT_DIR, DEFAULT_TSV_PATH, fingerprint_labels
from synthonor import build_synthon_fingerprint_cache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build packed synthon fingerprint caches for the bundled 1M synthon slice.")
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV_PATH, help="Bundled synthon TSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CACHE_OUTPUT_DIR,
        help="Directory for the cache build summary JSON.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild caches even if matching cache files already exist.")
    parser.add_argument("--chunk-size", type=int, default=25_000, help="Rows per fingerprinting chunk.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers passed to rdfp.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    builds: list[dict[str, object]] = []
    for spec in BIT_FINGERPRINT_SPECS:
        active_spec = spec.normalized()
        fp_started = time.perf_counter()
        cache_info = build_synthon_fingerprint_cache(
            args.tsv,
            fingerprint_spec=active_spec,
            overwrite=bool(args.overwrite),
            chunk_size=int(args.chunk_size),
            n_jobs=int(args.n_jobs),
        )
        builds.append(
            {
                "fingerprint": active_spec.label,
                "cache_prefix": str(cache_info.cache_prefix),
                "meta_path": str(cache_info.meta_path),
                "packed_fps_path": str(cache_info.packed_fps_path),
                "bitcounts_path": str(cache_info.bitcounts_path),
                "num_rows": int(cache_info.num_rows),
                "num_bytes": int(cache_info.num_bytes),
                "wall_time_sec": float(time.perf_counter() - fp_started),
            }
        )

    summary = {
        "tsv_path": str(args.tsv.resolve()),
        "fingerprints": fingerprint_labels(BIT_FINGERPRINT_SPECS),
        "builds": builds,
        "wall_time_sec": float(time.perf_counter() - started),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote cache summary to {summary_path}")
    for row in builds:
        print(
            f"{row['fingerprint']}: rows={row['num_rows']} bytes={row['num_bytes']} "
            f"wall={row['wall_time_sec']:.3f}s"
        )


if __name__ == "__main__":
    main()
