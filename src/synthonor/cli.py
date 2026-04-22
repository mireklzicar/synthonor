from __future__ import annotations

import argparse
import json
from typing import Sequence

from synthonor import __version__
from synthonor.fingerprints import FingerprintSpec
from synthonor.synthon_or_rdkit import (
    SearchConfig,
    build_synthon_fingerprint_cache,
    load_synthon_or_index,
    search_fingerprint,
    search_smiles,
    self_test_query_from_first_tuple,
    write_hits_jsonl,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthonor",
        description="Minimal standalone RDKit synthon-OR search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  synthonor data/syntons.tsv --build-cache-only\n"
            "  synthonor data/syntons.tsv --query 'CCN' --top-n 25 --output hits.jsonl\n"
            "  synthonor data/syntons.tsv --test --preset fast --top-n 5"
        ),
    )
    parser.add_argument("space_tsv", help="Path to a tab-separated synthon table.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--query", help="Query molecule as SMILES.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the built-in self-test query derived from the first complete tuple.",
    )
    parser.add_argument("--output", default="synthonor_hits.jsonl", help="Output JSONL path for hits.")
    parser.add_argument(
        "--skip-bad-rows",
        action="store_true",
        help="Skip malformed or unparsable synthon rows instead of raising.",
    )
    parser.add_argument(
        "--build-cache-only",
        action="store_true",
        help="Build or validate the packed synthon cache and exit.",
    )
    parser.add_argument("--cache-prefix", help="Cache file prefix (without extension).")
    parser.add_argument("--cache-dir", help="Directory used for auto-generated cache prefixes.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force cache rebuild even if cache exists.")
    parser.add_argument(
        "--cache-chunk-size",
        type=int,
        default=25_000,
        help="Rows per cache-build chunk. Lower values reduce peak memory usage.",
    )
    parser.add_argument(
        "--cache-n-jobs",
        type=int,
        default=1,
        help="Workers per cache-build chunk. Keep this at 1 on smaller machines.",
    )

    parser.add_argument("--fingerprint", default="ecfp4", help="Fingerprint family (for example: ecfp4, ecfp6).")
    parser.add_argument("--fp-size", type=int, default=2048, help="Fingerprint bit length.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius for Morgan-style fingerprints.")
    parser.add_argument(
        "--include-chirality",
        action="store_true",
        help="Include chirality information when supported by the fingerprint.",
    )
    parser.add_argument(
        "--use-features",
        action="store_true",
        help="Use feature invariants for Morgan fingerprints.",
    )
    parser.add_argument(
        "--union-mode",
        default="or",
        choices=["or", "sum", "max"],
        help="How to combine tuple fingerprints (count fingerprints only support sum/max).",
    )

    parser.add_argument(
        "--preset",
        default="accurate",
        choices=["fast", "accurate", "very_accurate"],
        help="Search preset controlling reaction routing and tuple limits.",
    )
    parser.add_argument("--reaction-cap", type=int, default=None, help="Maximum number of reactions to expand.")
    parser.add_argument(
        "--candidate-bank-size",
        type=int,
        default=None,
        help="Per-slot candidate bank size after prescreening.",
    )
    parser.add_argument(
        "--exhaustive-tuple-limit",
        type=int,
        default=None,
        help="Upper bound on fully enumerated tuples per reaction.",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Return at most this many hits.")
    parser.add_argument("--min-score", type=float, default=None, help="Keep only hits with score >= MIN_SCORE.")
    parser.add_argument("--max-score", type=float, default=None, help="Keep only hits with score <= MAX_SCORE.")
    return parser


def _fingerprint_spec_from_args(args: argparse.Namespace) -> FingerprintSpec:
    return FingerprintSpec(
        kind=str(args.fingerprint),
        fp_size=int(args.fp_size),
        radius=int(args.radius),
        include_chirality=bool(args.include_chirality),
        use_features=bool(args.use_features),
        union_mode=str(args.union_mode),
    ).normalized()


def _search_config_from_args(args: argparse.Namespace) -> SearchConfig | None:
    overrides: dict[str, int | None] = {}
    if args.reaction_cap is not None:
        overrides["reaction_cap"] = int(args.reaction_cap)
    if args.candidate_bank_size is not None:
        overrides["candidate_bank_size"] = int(args.candidate_bank_size)
    if args.exhaustive_tuple_limit is not None:
        overrides["exhaustive_tuple_limit"] = int(args.exhaustive_tuple_limit)
    if args.top_n is not None:
        overrides["topk_products"] = int(args.top_n)
    if not overrides:
        return None
    return SearchConfig.preset(str(args.preset), **overrides)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    has_query = args.query is not None
    has_test = bool(args.test)

    if args.build_cache_only and (has_query or has_test):
        parser.error("--build-cache-only cannot be combined with --query or --test")
    if not args.build_cache_only and has_query == has_test:
        parser.error("Provide exactly one of --query or --test")

    spec = _fingerprint_spec_from_args(args)
    config = _search_config_from_args(args)
    cache_info = None
    if not spec.is_count_fingerprint:
        cache_info = build_synthon_fingerprint_cache(
            args.space_tsv,
            fingerprint_spec=spec,
            skip_bad_rows=bool(args.skip_bad_rows),
            cache_prefix=args.cache_prefix,
            cache_dir=args.cache_dir,
            overwrite=bool(args.rebuild_cache),
            chunk_size=int(args.cache_chunk_size),
            n_jobs=int(args.cache_n_jobs),
        )
        if args.build_cache_only:
            print(
                json.dumps(
                    {
                        "fingerprint": spec.label,
                        "cache_prefix": cache_info.cache_prefix,
                        "meta_path": cache_info.meta_path,
                        "packed_fps_path": cache_info.packed_fps_path,
                        "bitcounts_path": cache_info.bitcounts_path,
                        "num_rows": cache_info.num_rows,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    elif args.build_cache_only:
        parser.error("--build-cache-only only supports bit fingerprints, not morgan_counts")

    index = load_synthon_or_index(
        args.space_tsv,
        fingerprint_spec=spec,
        skip_bad_rows=bool(args.skip_bad_rows),
        cache_prefix=args.cache_prefix,
        cache_dir=args.cache_dir,
        overwrite_cache=bool(args.rebuild_cache),
        cache_chunk_size=int(args.cache_chunk_size),
        cache_n_jobs=int(args.cache_n_jobs),
    )
    if has_query:
        hits = search_smiles(
            str(args.query),
            index,
            top_n=args.top_n,
            min_score=args.min_score,
            max_score=args.max_score,
            preset=str(args.preset),
            config=config,
        )
        query_label = str(args.query)
    else:
        query_fp = self_test_query_from_first_tuple(index)
        hits = search_fingerprint(
            query_fp,
            index,
            top_n=args.top_n,
            min_score=args.min_score,
            max_score=args.max_score,
            preset=str(args.preset),
            config=config,
        )
        query_label = "<self-test OR fingerprint of first tuple>"

    write_hits_jsonl(hits, args.output)
    print(
        json.dumps(
            {
                "query": query_label,
                "output": str(args.output),
                "num_hits": len(hits),
                "fingerprint": spec.label,
                "cache_prefix": index.fingerprint_cache.cache_prefix if index.fingerprint_cache is not None else None,
                "top_hit": hits[0].to_json_dict() if hits else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
