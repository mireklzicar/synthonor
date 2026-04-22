#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from _common import DEFAULT_BENCHMARK_OUTPUT_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot count-matched exact-similarity distributions for ground truth vs SynthonOR benchmark hits."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_OUTPUT_DIR,
        help="Directory containing summary.json, exact_ground_truth_top100.csv, and approx_retrievals.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plot PNGs and matched CSV exports. Defaults to <results-dir>/plots.",
    )
    parser.add_argument("--bins", type=int, default=36, help="Histogram bin count.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG dpi.")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{str(key): str(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _silverman_bandwidth(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.05
    std = float(np.std(values, ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    sigma = min(std, iqr / 1.34) if iqr > 0.0 else std
    if sigma <= 0.0:
        sigma = max(std, 0.05)
    return max(0.9 * sigma * (values.size ** (-1.0 / 5.0)), 0.02)


def _gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(grid)
    bandwidth = _silverman_bandwidth(values)
    deltas = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * deltas * deltas) / np.sqrt(2.0 * np.pi)
    density = kernel.mean(axis=1) / bandwidth
    return density


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_histogram(
    rows: list[dict[str, object]],
    *,
    fingerprint: str,
    output_path: Path,
    bins: int,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 4))
    palette = {
        "ground_truth": "#1f77b4",
        fingerprint: "#d62728",
    }
    grid = np.linspace(0.0, 1.0, 512)

    for label in ("ground_truth", fingerprint):
        scores = np.asarray(
            [float(row["tanimoto_similarity"]) for row in rows if str(row["method"]) == label],
            dtype=np.float64,
        )
        if scores.size == 0:
            continue
        plt.hist(
            scores,
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            histtype="step",
            linewidth=1.4,
            color=palette[label],
            label=label,
        )
        plt.plot(grid, _gaussian_kde(scores, grid), color=palette[label], linewidth=1.6)

    plt.xlabel("Tanimoto Similarity (Morgan 2048bit)")
    plt.ylabel("Density")
    plt.title(f"{fingerprint} - Count-matched results")
    plt.xlim(0.0, 1.0)
    plt.legend(title="Method")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or (results_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    truth_rows = _read_csv(results_dir / "exact_ground_truth_top100.csv")
    approx_rows = _read_csv(results_dir / "approx_retrievals.csv")

    truth_by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    approx_by_fingerprint_query: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in truth_rows:
        truth_by_query[str(row["query_id"])].append(row)
    for row in approx_rows:
        approx_by_fingerprint_query[(str(row["fingerprint"]), str(row["query_id"]))].append(row)

    manifest_rows: list[dict[str, object]] = []
    for method in summary["methods"]:
        fingerprint = str(method["fingerprint"])
        matched_rows: list[dict[str, object]] = []

        for query_id, truth_subset in truth_by_query.items():
            approx_subset = approx_by_fingerprint_query.get((fingerprint, query_id), [])
            if not approx_subset:
                continue

            truth_subset_sorted = sorted(truth_subset, key=lambda row: int(row["rank"]))
            approx_subset_sorted = sorted(approx_subset, key=lambda row: int(row["product_rank_unique"]))
            keep_n = min(len(truth_subset_sorted), len(approx_subset_sorted))
            if keep_n <= 0:
                continue

            for row in truth_subset_sorted[:keep_n]:
                matched_rows.append(
                    {
                        "query_id": int(row["query_id"]),
                        "query_smiles": str(row["query_smiles"]),
                        "fingerprint": fingerprint,
                        "method": "ground_truth",
                        "tanimoto_similarity": float(row["similarity"]),
                    }
                )
            for row in approx_subset_sorted[:keep_n]:
                matched_rows.append(
                    {
                        "query_id": int(row["query_id"]),
                        "query_smiles": str(row["query_smiles"]),
                        "fingerprint": fingerprint,
                        "method": fingerprint,
                        "tanimoto_similarity": float(row["exact_similarity"]),
                    }
                )

        if not matched_rows:
            continue

        csv_path = output_dir / f"{fingerprint}_count_matched_distribution.csv"
        png_path = output_dir / f"{fingerprint}_count_matched_hist_kde.png"
        _write_csv(csv_path, matched_rows)
        _make_histogram(matched_rows, fingerprint=fingerprint, output_path=png_path, bins=int(args.bins), dpi=int(args.dpi))

        manifest_rows.append(
            {
                "fingerprint": fingerprint,
                "rows": len(matched_rows),
                "csv_path": str(csv_path),
                "png_path": str(png_path),
            }
        )
        print(f"Wrote {png_path}")

    _write_csv(output_dir / "plot_manifest.csv", manifest_rows)


if __name__ == "__main__":
    main()
