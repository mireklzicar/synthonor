#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem

from _common import (
    ALL_FINGERPRINT_SPECS,
    BENCHMARK_FINGERPRINT_SPECS,
    DEFAULT_BENCHMARK_OUTPUT_DIR,
    DEFAULT_REACTIONS_PATH,
    DEFAULT_TSV_PATH,
    EXACT_FINGERPRINT_SPEC,
)
from synthonor import FingerprintSpec, SearchConfig, load_synthon_or_index, search_smiles
from synthonor.fingerprints import fingerprint_mol, fingerprint_smiles, tanimoto_similarity


@dataclass(frozen=True)
class SynthonRow:
    row_index: int
    reaction_id: str
    slot: int
    synthon_id: str
    smiles: str


@dataclass(frozen=True)
class QueryCase:
    query_id: int
    reaction_id: str
    row_ids: tuple[int, ...]
    query_smiles: str
    exact_fp: object
    truth_top100_smiles: frozenset[str]


@dataclass(frozen=True)
class ReactionSchema:
    reaction_id: str
    arity: int
    reaction_smarts: str


def load_reaction_schemas(path: str | Path) -> dict[str, ReactionSchema]:
    schemas: dict[str, ReactionSchema] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            reaction_id = str(row["reaction_id"]).strip()
            schemas[reaction_id] = ReactionSchema(
                reaction_id=reaction_id,
                arity=int(row["components"]),
                reaction_smarts=str(row["Reaction"]).strip(),
            )
    return schemas


@lru_cache(maxsize=512)
def _reaction_from_smarts(smarts: str):
    rxn = AllChem.ReactionFromSmarts(str(smarts))
    if rxn is None:
        raise ValueError(f"RDKit failed to parse reaction SMARTS: {smarts!r}")
    rxn.Initialize()
    return rxn


def _sanitize_with_fallback(mol: Chem.Mol) -> Chem.Mol:
    try:
        log_block = rdBase.BlockLogs()
        try:
            Chem.SanitizeMol(mol)
        finally:
            del log_block
    except Chem.KekulizeException:
        sanitize_ops = int(Chem.SanitizeFlags.SANITIZE_ALL) & ~int(Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
    return mol


def _assemble_product_from_reaction(
    schema: ReactionSchema,
    component_smiles: Sequence[str],
) -> tuple[Chem.Mol, str]:
    if len(component_smiles) != int(schema.arity):
        raise ValueError(
            f"Reaction {schema.reaction_id!r} expects {schema.arity} reactants, got {len(component_smiles)}"
        )
    reactants: list[Chem.Mol] = []
    for smiles in component_smiles:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"RDKit failed to parse synthon SMILES: {smiles!r}")
        reactants.append(mol)

    product_sets = _reaction_from_smarts(schema.reaction_smarts).RunReactants(tuple(reactants))
    if not product_sets:
        raise ValueError(f"Reaction {schema.reaction_id!r} did not produce products")

    best_mol: Chem.Mol | None = None
    best_smiles: str | None = None
    for product_set in product_sets:
        for product in product_set:
            candidate = _sanitize_with_fallback(Chem.Mol(product))
            smiles = Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
            if best_smiles is None or smiles < best_smiles:
                best_smiles = smiles
                best_mol = candidate
    if best_mol is None or best_smiles is None:
        raise ValueError(f"Reaction {schema.reaction_id!r} produced empty product tuples")
    return best_mol, best_smiles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SynthonOR fingerprint families against exact full-product retrieval on the bundled synthon_space_1M slice."
    )
    parser.add_argument("--tsv-space", type=Path, default=DEFAULT_TSV_PATH, help="Bundled synthon TSV path.")
    parser.add_argument(
        "--reactions",
        type=Path,
        default=DEFAULT_REACTIONS_PATH,
        help="Reaction schema table that matches the bundled TSV.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BENCHMARK_OUTPUT_DIR, help="Benchmark output directory.")
    parser.add_argument("--query-count", type=int, default=42, help="Number of deterministic query products to benchmark.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed used for per-reaction query sampling.")
    parser.add_argument(
        "--fingerprints",
        nargs="*",
        default=None,
        help="Optional fingerprint labels to benchmark. Defaults to the cacheable bit fingerprints used by the cache-build script.",
    )
    parser.add_argument(
        "--reaction-cap",
        type=str,
        default="8",
        help="SearchConfig.reaction_cap. Use an integer or 'none' to search all prescreened reactions.",
    )
    parser.add_argument("--candidate-bank-size", type=int, default=64, help="SearchConfig.candidate_bank_size.")
    parser.add_argument("--exhaustive-tuple-limit", type=int, default=50_000, help="SearchConfig.exhaustive_tuple_limit.")
    parser.add_argument("--topk-products", type=int, default=100, help="SearchConfig.topk_products.")
    parser.add_argument("--cache-chunk-size", type=int, default=25_000, help="Rows per fingerprint cache chunk for bit fingerprints.")
    parser.add_argument("--cache-n-jobs", type=int, default=1, help="rdfp worker count for bit fingerprint caches.")
    return parser.parse_args()


def _parse_reaction_cap(value: object) -> int | None:
    text = str(value).strip().lower()
    if text in {"none", "all", "null"}:
        return None
    return int(text)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    idx = max(min(int(np.ceil(float(fraction) * len(ordered))) - 1, len(ordered) - 1), 0)
    return float(ordered[idx])


def _bitvect_to_packed(fp: object) -> tuple[np.ndarray, int]:
    packed = np.frombuffer(DataStructs.BitVectToBinaryText(fp), dtype=np.uint8)  # type: ignore[arg-type]
    return packed, int(fp.GetNumOnBits())  # type: ignore[union-attr]


_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


def _exact_scores_against_library(
    query_packed: np.ndarray,
    query_bit_count: int,
    library_packed: np.ndarray,
    library_bits: np.ndarray,
) -> np.ndarray:
    intersections = _POPCOUNT_LUT[np.bitwise_and(library_packed, query_packed[None, :])].sum(axis=1).astype(np.float32, copy=False)
    unions = np.maximum(float(query_bit_count) + library_bits.astype(np.float32, copy=False) - intersections, 1.0)
    return intersections / unions


def _read_synthon_rows(tsv_path: Path) -> tuple[list[SynthonRow], dict[str, dict[int, list[SynthonRow]]]]:
    rows: list[SynthonRow] = []
    by_reaction: dict[str, dict[int, list[SynthonRow]]] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_index, row in enumerate(reader):
            item = SynthonRow(
                row_index=int(row_index),
                reaction_id=str(row["reaction_id"]).strip(),
                slot=int(row["position"]),
                synthon_id=str(row["synthon_id"]).strip(),
                smiles=str(row["smiles"]).strip(),
            )
            rows.append(item)
            by_reaction.setdefault(item.reaction_id, {}).setdefault(item.slot, []).append(item)
    return rows, by_reaction


def _iter_product_combos(
    by_reaction: dict[str, dict[int, list[SynthonRow]]],
    schemas: dict[str, ReactionSchema],
) -> Iterable[tuple[str, tuple[SynthonRow, ...], ReactionSchema]]:
    for reaction_id in sorted(by_reaction):
        schema = schemas[reaction_id]
        role_ids = sorted(by_reaction[reaction_id])
        pools = [by_reaction[reaction_id][slot] for slot in role_ids]
        if len(pools) != int(schema.arity):
            continue
        for combo in product(*pools):
            yield reaction_id, tuple(combo), schema


def build_exact_library(
    *,
    tsv_path: Path,
    reactions_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    rows, by_reaction = _read_synthon_rows(tsv_path)
    schemas = load_reaction_schemas(reactions_path)

    library_dir = output_dir / "exact_library"
    library_dir.mkdir(parents=True, exist_ok=True)
    summary_path = library_dir / "exact_product_library_summary.json"
    smiles_path = library_dir / "exact_product_smiles.csv"
    packed_path = library_dir / "exact_product_fingerprints.u8"
    bits_path = library_dir / "exact_product_bitcounts.u32"

    if summary_path.exists() and smiles_path.exists() and packed_path.exists() and bits_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        num_products = int(summary["unique_product_count"])
        num_bytes = int(summary["num_bytes"])
        packed_matrix = np.memmap(packed_path, dtype=np.uint8, mode="r", shape=(num_products, num_bytes))
        bit_array = np.memmap(bits_path, dtype=np.uint32, mode="r", shape=(num_products,))
        smiles_to_index: dict[str, int] = {}
        with smiles_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                smiles_to_index[str(row["product_smiles"])] = int(row["product_index"])
        return {
            "summary": summary,
            "schemas": schemas,
            "by_reaction": by_reaction,
            "product_packed": packed_matrix,
            "product_bits": bit_array,
            "smiles_to_index": smiles_to_index,
        }

    tmp_smiles = library_dir / "exact_product_smiles.csv.tmp"
    tmp_packed = library_dir / "exact_product_fingerprints.u8.tmp"
    tmp_bits = library_dir / "exact_product_bitcounts.u32.tmp"

    unique_smiles_to_index: dict[str, int] = {}
    total_product_count = 0
    duplicate_product_count = 0
    failed_assembly_count = 0
    num_bytes = 0
    started = time.perf_counter()

    try:
        with (
            tmp_smiles.open("w", encoding="utf-8", newline="") as smiles_handle,
            tmp_packed.open("wb") as packed_handle,
            tmp_bits.open("wb") as bits_handle,
        ):
            writer = csv.DictWriter(
                smiles_handle,
                fieldnames=["product_index", "reaction_id", "row_ids", "product_smiles"],
            )
            writer.writeheader()

            for reaction_id, combo, schema in _iter_product_combos(by_reaction, schemas):
                total_product_count += 1
                try:
                    product_mol, product_smiles = _assemble_product_from_reaction(
                        schema,
                        [row.smiles for row in combo],
                    )
                except Exception:
                    failed_assembly_count += 1
                    continue
                if product_smiles in unique_smiles_to_index:
                    duplicate_product_count += 1
                    continue
                product_fp = fingerprint_mol(product_mol, EXACT_FINGERPRINT_SPEC)
                packed_fp, bit_count = _bitvect_to_packed(product_fp)
                if num_bytes == 0:
                    num_bytes = int(packed_fp.shape[0])
                unique_index = len(unique_smiles_to_index)
                unique_smiles_to_index[product_smiles] = unique_index
                writer.writerow(
                    {
                        "product_index": unique_index,
                        "reaction_id": reaction_id,
                        "row_ids": ",".join(str(row.row_index) for row in combo),
                        "product_smiles": product_smiles,
                    }
                )
                packed_handle.write(bytes(packed_fp))
                bits_handle.write(np.asarray([bit_count], dtype=np.uint32).tobytes())

        summary = {
            "tsv_path": str(tsv_path.resolve()),
            "reactions_path": str(reactions_path.resolve()),
            "total_product_count": int(total_product_count),
            "duplicate_product_count": int(duplicate_product_count),
            "failed_assembly_count": int(failed_assembly_count),
            "unique_product_count": int(len(unique_smiles_to_index)),
            "num_bytes": int(num_bytes),
            "wall_time_sec": float(time.perf_counter() - started),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        tmp_smiles.replace(smiles_path)
        tmp_packed.replace(packed_path)
        tmp_bits.replace(bits_path)
    finally:
        for leftover in (tmp_smiles, tmp_packed, tmp_bits):
            if leftover.exists():
                leftover.unlink()

    packed_matrix = np.memmap(packed_path, dtype=np.uint8, mode="r", shape=(len(unique_smiles_to_index), num_bytes))
    bit_array = np.memmap(bits_path, dtype=np.uint32, mode="r", shape=(len(unique_smiles_to_index),))
    return {
        "summary": summary,
        "schemas": schemas,
        "by_reaction": by_reaction,
        "product_packed": packed_matrix,
        "product_bits": bit_array,
        "smiles_to_index": unique_smiles_to_index,
    }


def _sample_queries(
    by_reaction: dict[str, dict[int, list[SynthonRow]]],
    schemas: dict[str, ReactionSchema],
    *,
    query_count: int,
    seed: int,
    library_packed: np.ndarray,
    library_bits: np.ndarray,
    truth_export: list[dict[str, object]],
) -> list[QueryCase]:
    rng = random.Random(int(seed))
    queries: list[QueryCase] = []
    for reaction_id in sorted(by_reaction):
        if len(queries) >= int(query_count):
            break
        pools = by_reaction[reaction_id]
        role_ids = sorted(pools)
        selected_rows = tuple(pools[slot][rng.randrange(len(pools[slot]))] for slot in role_ids)
        _, query_smiles = _assemble_product_from_reaction(
            schemas[reaction_id],
            [row.smiles for row in selected_rows],
        )
        query_fp = fingerprint_smiles(query_smiles, EXACT_FINGERPRINT_SPEC)
        query_packed, query_bit_count = _bitvect_to_packed(query_fp)
        exact_scores = _exact_scores_against_library(query_packed, query_bit_count, library_packed, library_bits)
        top100_indices = np.argsort(-exact_scores, kind="stable")[:100]
        truth_top100_smiles: list[str] = []
        for rank, product_index in enumerate(top100_indices, start=1):
            truth_export.append(
                {
                    "query_id": int(len(queries)),
                    "query_reaction_id": reaction_id,
                    "query_smiles": query_smiles,
                    "rank": int(rank),
                    "product_index": int(product_index),
                    "similarity": float(exact_scores[int(product_index)]),
                }
            )
            truth_top100_smiles.append("")
        queries.append(
            QueryCase(
                query_id=int(len(queries)),
                reaction_id=reaction_id,
                row_ids=tuple(int(row.row_index) for row in selected_rows),
                query_smiles=query_smiles,
                exact_fp=query_fp,
                truth_top100_smiles=frozenset(),
            )
        )
    return queries


def _load_truth_smiles(exact_smiles_csv: Path, truth_rows: list[dict[str, object]]) -> list[QueryCase]:
    product_smiles_by_index: dict[int, str] = {}
    with exact_smiles_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            product_smiles_by_index[int(row["product_index"])] = str(row["product_smiles"])

    grouped: dict[int, list[dict[str, object]]] = {}
    for row in truth_rows:
        grouped.setdefault(int(row["query_id"]), []).append(row)

    queries: list[QueryCase] = []
    for query_id in sorted(grouped):
        rows = sorted(grouped[query_id], key=lambda item: int(item["rank"]))
        query_smiles = str(rows[0]["query_smiles"])
        query_fp = fingerprint_smiles(query_smiles, EXACT_FINGERPRINT_SPEC)
        truth_smiles = frozenset(product_smiles_by_index[int(row["product_index"])] for row in rows)
        queries.append(
            QueryCase(
                query_id=int(rows[0]["query_id"]),
                reaction_id=str(rows[0]["query_reaction_id"]),
                row_ids=tuple(),
                query_smiles=query_smiles,
                exact_fp=query_fp,
                truth_top100_smiles=truth_smiles,
            )
        )
    return queries


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


def _selected_specs(labels: Sequence[str] | None) -> list[FingerprintSpec]:
    spec_by_label = {spec.normalized().label: spec for spec in ALL_FINGERPRINT_SPECS}
    if not labels:
        return list(BENCHMARK_FINGERPRINT_SPECS)
    chosen: list[FingerprintSpec] = []
    missing: list[str] = []
    for label in labels:
        normalized = str(label).strip().lower()
        if normalized not in spec_by_label:
            missing.append(normalized)
            continue
        chosen.append(spec_by_label[normalized])
    if missing:
        raise ValueError(f"Unknown fingerprint labels: {missing!r}")
    return chosen


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    selected_specs = _selected_specs(args.fingerprints)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_library = build_exact_library(
        tsv_path=args.tsv_space,
        reactions_path=args.reactions,
        output_dir=output_dir,
    )
    schemas = exact_library["schemas"]
    by_reaction = exact_library["by_reaction"]
    product_packed = exact_library["product_packed"]
    product_bits = exact_library["product_bits"]

    truth_rows: list[dict[str, object]] = []
    sampled_queries = _sample_queries(
        by_reaction,
        schemas,
        query_count=int(args.query_count),
        seed=int(args.seed),
        library_packed=product_packed,
        library_bits=product_bits,
        truth_export=truth_rows,
    )

    exact_smiles_csv = output_dir / "exact_library" / "exact_product_smiles.csv"
    product_smiles_by_index: dict[int, str] = {}
    with exact_smiles_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            product_smiles_by_index[int(row["product_index"])] = str(row["product_smiles"])
    for row in truth_rows:
        row["product_smiles"] = product_smiles_by_index[int(row["product_index"])]

    query_lookup = {query.query_id: query for query in sampled_queries}
    enriched_queries = _load_truth_smiles(exact_smiles_csv, truth_rows)
    query_lookup = {query.query_id: query for query in enriched_queries}

    approx_retrieval_rows: list[dict[str, object]] = []
    method_summaries: list[dict[str, object]] = []
    active_config = SearchConfig(
        reaction_cap=_parse_reaction_cap(args.reaction_cap),
        candidate_bank_size=int(args.candidate_bank_size),
        exhaustive_tuple_limit=int(args.exhaustive_tuple_limit),
        topk_products=int(args.topk_products),
    )

    for spec in selected_specs:
        active_spec = spec.normalized()
        index = load_synthon_or_index(
            args.tsv_space,
            fingerprint_spec=active_spec,
            cache_chunk_size=int(args.cache_chunk_size),
            cache_n_jobs=int(args.cache_n_jobs),
        )
        query_metrics: list[dict[str, object]] = []

        for query_id in sorted(query_lookup):
            query = query_lookup[query_id]
            started = time.perf_counter()
            hits = search_smiles(query.query_smiles, index, config=active_config)
            wall_time = float(time.perf_counter() - started)

            seen_smiles: set[str] = set()
            exact_scores: list[float] = []
            overlap_count = 0
            invalid_assemblies = 0
            unique_rank = 0
            for hit in hits:
                schema = schemas.get(str(hit.reaction_id))
                if schema is None:
                    continue
                try:
                    _, product_smiles = _assemble_product_from_reaction(schema, hit.smiles)
                except Exception:
                    invalid_assemblies += 1
                    continue
                if product_smiles in seen_smiles:
                    continue
                seen_smiles.add(product_smiles)
                unique_rank += 1
                exact_similarity = float(
                    tanimoto_similarity(
                        query.exact_fp,  # type: ignore[arg-type]
                        fingerprint_smiles(product_smiles, EXACT_FINGERPRINT_SPEC),
                        EXACT_FINGERPRINT_SPEC,
                    )
                )
                in_truth = int(product_smiles in query.truth_top100_smiles)
                overlap_count += in_truth
                exact_scores.append(exact_similarity)
                approx_retrieval_rows.append(
                    {
                        "query_id": int(query.query_id),
                        "query_reaction_id": str(query.reaction_id),
                        "query_smiles": str(query.query_smiles),
                        "fingerprint": active_spec.label,
                        "approx_rank": int(hit.rank),
                        "product_rank_unique": int(unique_rank),
                        "product_smiles": product_smiles,
                        "exact_similarity": exact_similarity,
                        "in_exact_truth_top100": in_truth,
                    }
                )

            query_metrics.append(
                {
                    "query_id": int(query.query_id),
                    "query_reaction_id": str(query.reaction_id),
                    "query_smiles": str(query.query_smiles),
                    "wall_time_sec": wall_time,
                    "retrieved_unique_products": int(len(exact_scores)),
                    "truth_top100_overlap": int(overlap_count),
                    "truth_top100_recall": float(overlap_count / 100.0),
                    "exact_sum_top100_similarity": float(sum(exact_scores)),
                    "exact_mean_top100_similarity": float(sum(exact_scores) / max(len(exact_scores), 1)) if exact_scores else 0.0,
                    "exact_top1_similarity": float(exact_scores[0]) if exact_scores else 0.0,
                    "invalid_assemblies": int(invalid_assemblies),
                }
            )

        wall_times = [float(row["wall_time_sec"]) for row in query_metrics]
        method_summaries.append(
            {
                "method": "synthonor",
                "fingerprint": active_spec.label,
                "uses_packed_cache": bool(index.uses_packed_fingerprints),
                "queries": query_metrics,
                "mean_wall_time_sec": float(statistics.mean(wall_times)) if wall_times else 0.0,
                "median_wall_time_sec": float(statistics.median(wall_times)) if wall_times else 0.0,
                "p95_wall_time_sec": _percentile(wall_times, 0.95),
                "mean_retrieved_unique_products": float(
                    statistics.mean(float(row["retrieved_unique_products"]) for row in query_metrics)
                )
                if query_metrics
                else 0.0,
                "mean_truth_top100_overlap": float(
                    statistics.mean(float(row["truth_top100_overlap"]) for row in query_metrics)
                )
                if query_metrics
                else 0.0,
                "mean_truth_top100_recall": float(
                    statistics.mean(float(row["truth_top100_recall"]) for row in query_metrics)
                )
                if query_metrics
                else 0.0,
                "mean_exact_sum_top100_similarity": float(
                    statistics.mean(float(row["exact_sum_top100_similarity"]) for row in query_metrics)
                )
                if query_metrics
                else 0.0,
                "mean_exact_mean_top100_similarity": float(
                    statistics.mean(float(row["exact_mean_top100_similarity"]) for row in query_metrics)
                )
                if query_metrics
                else 0.0,
                "mean_exact_top1_similarity": float(
                    statistics.mean(float(row["exact_top1_similarity"]) for row in query_metrics)
                )
                if query_metrics
                else 0.0,
            }
        )

    summary = {
        "query_count": int(len(query_lookup)),
        "seed": int(args.seed),
        "query_definition": {
            "sampling": "one_random_tuple_per_reaction",
            "tsv_space": str(args.tsv_space.resolve()),
            "reactions": str(args.reactions.resolve()),
        },
        "search_config": {
            "reaction_cap": active_config.reaction_cap,
            "candidate_bank_size": int(active_config.candidate_bank_size),
            "exhaustive_tuple_limit": int(active_config.exhaustive_tuple_limit),
            "topk_products": int(active_config.topk_products),
        },
        "exact_library": exact_library["summary"],
        "methods": method_summaries,
    }

    _write_csv(output_dir / "exact_ground_truth_top100.csv", truth_rows)
    _write_csv(output_dir / "approx_retrievals.csv", approx_retrieval_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_benchmark(args)
    print(f"Wrote benchmark summary to {args.output_dir / 'summary.json'}")
    for method in summary["methods"]:
        print(
            f"{method['fingerprint']}: overlap={method['mean_truth_top100_overlap']:.2f} "
            f"top1={method['mean_exact_top1_similarity']:.4f} wall={method['mean_wall_time_sec']:.3f}s"
        )


if __name__ == "__main__":
    main()
