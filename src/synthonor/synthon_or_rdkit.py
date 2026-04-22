from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from rdfp import smiles_to_fps
from rdkit import DataStructs

from synthonor.fingerprints import (
    FingerprintSpec,
    FingerprintValue,
    bulk_tanimoto_similarity,
    fingerprint_smiles,
    quiet_rdkit_logs,
    tanimoto_similarity,
    union_fingerprints,
)


SearchPreset = Literal["fast", "accurate", "very_accurate"]
_CACHE_VERSION = 2
_POPCOUNT_LUT = np.asarray([bin(value).count("1") for value in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class FingerprintCacheInfo:
    cache_prefix: str
    meta_path: str
    packed_fps_path: str
    bitcounts_path: str
    num_rows: int
    num_bytes: int
    fingerprint: str


@dataclass(frozen=True)
class SynthonRecord:
    row_index: int
    smiles: str
    synthon_id: str
    slot: int
    fp: FingerprintValue | None = None


@dataclass(frozen=True)
class SlotIndex:
    slot: int
    records: tuple[SynthonRecord, ...]

    @property
    def fps(self) -> tuple[FingerprintValue | None, ...]:
        return tuple(record.fp for record in self.records)

    @property
    def size(self) -> int:
        return len(self.records)

    def take(self, n: int) -> "SlotIndex":
        limit = max(0, min(int(n), self.size))
        return SlotIndex(slot=self.slot, records=self.records[:limit])


@dataclass(frozen=True)
class ReactionIndex:
    reaction_id: str
    slots: tuple[SlotIndex, ...]


@dataclass(frozen=True)
class SearchConfig:
    reaction_cap: int | None = 8
    candidate_bank_size: int = 192
    exhaustive_tuple_limit: int = 250_000
    topk_products: int = 100

    @classmethod
    def preset(
        cls,
        name: SearchPreset = "accurate",
        *,
        topk_products: int = 100,
        **overrides: object,
    ) -> "SearchConfig":
        label = _normalize_search_preset_name(name)
        if label == "fast":
            base = cls(
                reaction_cap=1,
                candidate_bank_size=64,
                exhaustive_tuple_limit=50_000,
                topk_products=int(topk_products),
            )
        elif label == "accurate":
            base = cls(
                reaction_cap=8,
                candidate_bank_size=192,
                exhaustive_tuple_limit=250_000,
                topk_products=int(topk_products),
            )
        elif label == "very_accurate":
            base = cls(
                reaction_cap=None,
                candidate_bank_size=256,
                exhaustive_tuple_limit=500_000,
                topk_products=int(topk_products),
            )
        else:
            raise ValueError(f"Unsupported search preset: {name!r}")
        return replace(base, **overrides)


@dataclass(frozen=True)
class ApproxHit:
    rank: int
    reaction_id: str
    approx_score: float
    synthon_ids: tuple[str, ...]
    slots: tuple[int, ...]
    smiles: tuple[str, ...]
    row_indices: tuple[int, ...]

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["approx_score"] = round(float(self.approx_score), 8)
        return data


@dataclass(frozen=True)
class MaterializedCandidate:
    approx_score: float
    reaction_id: str
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class SynthonORIndex:
    reactions: tuple[ReactionIndex, ...]
    fingerprint_spec: FingerprintSpec
    row_lookup: dict[int, SynthonRecord]
    packed_fingerprints: np.ndarray | None = None
    row_bitcounts: np.ndarray | None = None
    fingerprint_cache: FingerprintCacheInfo | None = None

    @property
    def num_reactions(self) -> int:
        return len(self.reactions)

    @property
    def num_synthons(self) -> int:
        return len(self.row_lookup)

    @property
    def num_slots(self) -> int:
        return sum(len(reaction.slots) for reaction in self.reactions)

    @property
    def uses_packed_fingerprints(self) -> bool:
        return self.packed_fingerprints is not None and self.row_bitcounts is not None


def _normalize_search_preset_name(name: SearchPreset | str) -> str:
    label = str(name).strip().lower().replace("-", "_")
    if label not in {"fast", "accurate", "very_accurate"}:
        raise ValueError(f"Unsupported search preset: {name!r}")
    return label


def _find_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(
        f"Missing required column. Tried {list(candidates)!r}; available columns: {list(fieldnames)!r}"
    )


def _cache_paths(cache_prefix: str | Path) -> dict[str, Path]:
    prefix = Path(cache_prefix)
    return {
        "prefix": prefix,
        "meta": Path(str(prefix) + ".meta.json"),
        "packed_fps": Path(str(prefix) + ".packed_fps.u8"),
        "bitcounts": Path(str(prefix) + ".bitcounts.u32"),
    }


def default_fingerprint_cache_prefix(
    tsv_path: str | Path,
    *,
    fingerprint_spec: FingerprintSpec | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    path = Path(tsv_path)
    spec = (fingerprint_spec or FingerprintSpec()).normalized()
    root = Path(cache_dir) if cache_dir is not None else path.parent
    return root / f"{path.stem}.{spec.label}.synthon_fp_cache"


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _source_signature(path: Path, *, include_hash: bool = False) -> dict[str, object]:
    stat = path.stat()
    signature: dict[str, object] = {
        "name": path.name,
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        signature["content_sha256"] = _file_sha256(path)
    return signature


def _source_signature_matches(
    meta_source: object,
    *,
    expected_source: dict[str, object],
    source_path: Path,
) -> bool:
    if not isinstance(meta_source, dict):
        return False
    meta_name = meta_source.get("name")
    if meta_name is None:
        meta_name = Path(str(meta_source.get("path", ""))).name
    if str(meta_name) != str(expected_source["name"]):
        return False
    if int(meta_source.get("size", -1)) != int(expected_source["size"]):
        return False
    if int(meta_source.get("mtime_ns", -1)) != int(expected_source["mtime_ns"]):
        return False

    expected_hash = str(meta_source.get("content_sha256", "")).strip().lower()
    if expected_hash:
        return expected_hash == _file_sha256(source_path).lower()
    return True


def _bitvect_to_packed_bytes(fp: FingerprintValue) -> bytes:
    return bytes(DataStructs.BitVectToBinaryText(fp))  # type: ignore[arg-type]


def _bitcount_from_packed_array(packed: np.ndarray) -> int:
    return int(_POPCOUNT_LUT[np.asarray(packed, dtype=np.uint8)].sum(dtype=np.uint32))


def _rdfp_fp_type_and_kwargs(spec: FingerprintSpec) -> tuple[str, dict[str, object]]:
    normalized = spec.normalized()
    if normalized.kind == "morgan":
        return "morgan", {"radius": int(normalized.radius)}
    if normalized.kind == "rdkit":
        return "rdkit", {}
    if normalized.kind == "patternfp":
        return "pattern", {}
    if normalized.kind == "atom_pair":
        return "atom_pair", {}
    if normalized.kind == "topological_torsion":
        return "topological_torsion", {}
    raise ValueError(f"Unsupported fingerprint kind for packed cache build: {normalized.kind!r}")


def _bitvect_to_packed_array(fp: FingerprintValue) -> np.ndarray:
    return np.frombuffer(_bitvect_to_packed_bytes(fp), dtype=np.uint8)


def _packed_array_to_bitvect(packed: np.ndarray):
    return DataStructs.CreateFromBinaryText(bytes(np.asarray(packed, dtype=np.uint8)))


def _packed_tanimoto_single(
    query_packed: np.ndarray,
    query_on_bits: int,
    target_packed: np.ndarray,
) -> float:
    intersection = int(_POPCOUNT_LUT[np.bitwise_and(query_packed, target_packed)].sum(dtype=np.uint32))
    target_on_bits = _bitcount_from_packed_array(target_packed)
    denom = int(query_on_bits + target_on_bits - intersection)
    if denom <= 0:
        return 1.0
    return float(intersection / denom)


def _bulk_tanimoto_similarity_packed(
    query_packed: np.ndarray,
    packed_fps: np.ndarray,
    row_bitcounts: np.ndarray,
    *,
    chunk_size: int = 65536,
) -> list[float]:
    total = int(packed_fps.shape[0])
    if total == 0:
        return []

    scores = np.zeros((total,), dtype=np.float32)
    query_on_bits = _bitcount_from_packed_array(query_packed)
    for start in range(0, total, max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), total)
        chunk = np.asarray(packed_fps[start:stop], dtype=np.uint8)
        intersections = _POPCOUNT_LUT[np.bitwise_and(chunk, query_packed)].sum(axis=1, dtype=np.uint32)
        target_counts = np.asarray(row_bitcounts[start:stop], dtype=np.uint32)
        denominators = query_on_bits + target_counts - intersections
        chunk_scores = np.divide(
            intersections,
            denominators,
            out=np.zeros((stop - start,), dtype=np.float32),
            where=denominators > 0,
        )
        if query_on_bits == 0:
            chunk_scores[target_counts == 0] = 1.0
        scores[start:stop] = chunk_scores
    return [float(value) for value in scores]


def _union_row_fingerprints(row_indices: Sequence[int], index: SynthonORIndex) -> FingerprintValue:
    if index.uses_packed_fingerprints:
        if index.packed_fingerprints is None:
            raise ValueError("Packed fingerprint storage is unavailable")
        combined = np.zeros((int(index.packed_fingerprints.shape[1]),), dtype=np.uint8)
        for row_idx in row_indices:
            np.bitwise_or(combined, np.asarray(index.packed_fingerprints[int(row_idx)], dtype=np.uint8), out=combined)
        return _packed_array_to_bitvect(combined)

    fps = []
    for row_idx in row_indices:
        fp = index.row_lookup[int(row_idx)].fp
        if fp is None:
            raise ValueError("Missing in-memory fingerprint for row")
        fps.append(fp)
    return union_fingerprints(fps, index.fingerprint_spec)


def build_synthon_fingerprint_cache(
    tsv_path: str | Path,
    *,
    fingerprint_spec: FingerprintSpec | None = None,
    quiet_logs: bool = True,
    skip_bad_rows: bool = False,
    cache_prefix: str | Path | None = None,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
    chunk_size: int = 25_000,
    n_jobs: int = 1,
) -> FingerprintCacheInfo:
    spec = (fingerprint_spec or FingerprintSpec()).normalized()
    if spec.is_count_fingerprint:
        raise ValueError("Packed synthon fingerprint caches only support bit fingerprints, not morgan_counts")
    quiet_rdkit_logs(quiet_logs)
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")

    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if cache_prefix is None:
        cache_prefix = default_fingerprint_cache_prefix(path, fingerprint_spec=spec, cache_dir=cache_dir)

    paths = _cache_paths(cache_prefix)
    expected_source = _source_signature(path, include_hash=False)
    expected_spec = asdict(spec)

    if not overwrite and paths["meta"].exists() and paths["packed_fps"].exists() and paths["bitcounts"].exists():
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if (
            int(meta.get("version", -1)) == _CACHE_VERSION
            and _source_signature_matches(
                meta.get("source"),
                expected_source=expected_source,
                source_path=path,
            )
            and meta.get("fingerprint_spec") == expected_spec
            and bool(meta.get("skip_bad_rows", False)) == bool(skip_bad_rows)
        ):
            return FingerprintCacheInfo(
                cache_prefix=str(paths["prefix"]),
                meta_path=str(paths["meta"]),
                packed_fps_path=str(paths["packed_fps"]),
                bitcounts_path=str(paths["bitcounts"]),
                num_rows=int(meta["num_rows"]),
                num_bytes=int(meta["num_bytes"]),
                fingerprint=str(spec.label),
            )

    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    tmp_suffix = f".tmp.{os.getpid()}"
    tmp_meta = Path(str(paths["meta"]) + tmp_suffix)
    tmp_packed = Path(str(paths["packed_fps"]) + tmp_suffix)
    tmp_counts = Path(str(paths["bitcounts"]) + tmp_suffix)
    rdfp_type, rdfp_kwargs = _rdfp_fp_type_and_kwargs(spec)

    num_rows = 0
    num_bytes: int | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle, tmp_packed.open("wb") as packed_handle, tmp_counts.open("wb") as counts_handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError(f"No header found in {path}")
            smiles_col = _find_column(reader.fieldnames, ["SMILES", "smiles"])
            slot_col = _find_column(reader.fieldnames, ["synthon#", "synton#", "slot", "position"])
            reaction_col = _find_column(reader.fieldnames, ["reaction_id", "rxn_id", "reaction"])
            chunk_smiles: list[str] = []
            chunk_lines: list[int] = []

            def flush_chunk() -> None:
                nonlocal num_rows, num_bytes, chunk_smiles, chunk_lines
                if not chunk_smiles:
                    return
                fps, invalid_mask = smiles_to_fps(
                    chunk_smiles,
                    fp_type=rdfp_type,
                    fp_size=int(spec.fp_size),
                    n_jobs=int(n_jobs),
                    # Some bundled synthons trip RDKit generator preconditions unless
                    # implicit valence/hydrogen state has been initialized first.
                    sanitize=True,
                    progress=False,
                    **rdfp_kwargs,
                )
                invalid_indices = np.flatnonzero(invalid_mask).tolist()
                if invalid_indices and not skip_bad_rows:
                    first_invalid = int(invalid_indices[0])
                    raise ValueError(
                        f"Failed to fingerprint row at line {chunk_lines[first_invalid]}: "
                        f"{chunk_smiles[first_invalid]!r}"
                    )

                if fps.shape[0]:
                    packed_chunk = np.packbits(fps.astype(np.uint8, copy=False), axis=1, bitorder="little")
                    bitcounts_chunk = np.asarray(fps.sum(axis=1, dtype=np.uint32), dtype=np.uint32)
                    if num_bytes is None:
                        num_bytes = int(packed_chunk.shape[1])
                    elif int(packed_chunk.shape[1]) != int(num_bytes):
                        raise ValueError(
                            f"Fingerprint byte width changed within one cache build: {packed_chunk.shape[1]} vs {num_bytes}"
                        )
                    packed_handle.write(packed_chunk.tobytes())
                    counts_handle.write(bitcounts_chunk.tobytes())
                    num_rows += int(packed_chunk.shape[0])

                chunk_smiles = []
                chunk_lines = []
                gc.collect()

            for row_number, row in enumerate(reader, start=2):
                smiles = str(row[smiles_col]).strip()
                slot_text = str(row[slot_col]).strip()
                reaction_id = str(row[reaction_col]).strip()
                if not smiles or not slot_text or not reaction_id:
                    if skip_bad_rows:
                        continue
                    raise ValueError(f"Missing SMILES/reaction/slot value at line {row_number}")
                try:
                    int(slot_text)
                except Exception as exc:
                    if skip_bad_rows:
                        continue
                    raise ValueError(f"Failed to fingerprint row at line {row_number}: {row!r}") from exc
                chunk_smiles.append(smiles)
                chunk_lines.append(int(row_number))
                if len(chunk_smiles) >= int(chunk_size):
                    flush_chunk()
            flush_chunk()

        final_num_bytes = int(num_bytes or 0)
        source_signature = _source_signature(path, include_hash=True)
        meta = {
            "version": _CACHE_VERSION,
            "source": source_signature,
            "fingerprint_spec": expected_spec,
            "skip_bad_rows": bool(skip_bad_rows),
            "num_rows": int(num_rows),
            "num_bytes": final_num_bytes,
            "fingerprint": spec.label,
            "paths": {
                "packed_fps": str(paths["packed_fps"]),
                "bitcounts": str(paths["bitcounts"]),
            },
        }
        tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_packed, paths["packed_fps"])
        os.replace(tmp_counts, paths["bitcounts"])
        os.replace(tmp_meta, paths["meta"])
    finally:
        for leftover in [tmp_meta, tmp_packed, tmp_counts]:
            if leftover.exists():
                leftover.unlink()

    return FingerprintCacheInfo(
        cache_prefix=str(paths["prefix"]),
        meta_path=str(paths["meta"]),
        packed_fps_path=str(paths["packed_fps"]),
        bitcounts_path=str(paths["bitcounts"]),
        num_rows=int(num_rows),
        num_bytes=int(num_bytes or 0),
        fingerprint=str(spec.label),
    )


def _open_packed_fingerprint_cache(cache_info: FingerprintCacheInfo) -> tuple[np.ndarray, np.ndarray]:
    packed = np.memmap(
        cache_info.packed_fps_path,
        dtype=np.uint8,
        mode="r",
        shape=(int(cache_info.num_rows), int(cache_info.num_bytes)),
    )
    counts = np.memmap(
        cache_info.bitcounts_path,
        dtype=np.uint32,
        mode="r",
        shape=(int(cache_info.num_rows),),
    )
    return packed, counts


def load_synthon_or_index(
    tsv_path: str | Path,
    *,
    fingerprint_spec: FingerprintSpec | None = None,
    quiet_logs: bool = True,
    skip_bad_rows: bool = False,
    cache_prefix: str | Path | None = None,
    cache_dir: str | Path | None = None,
    overwrite_cache: bool = False,
    cache_chunk_size: int = 25_000,
    cache_n_jobs: int = 1,
) -> SynthonORIndex:
    spec = (fingerprint_spec or FingerprintSpec()).normalized()
    quiet_rdkit_logs(quiet_logs)

    path = Path(tsv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    cache_info: FingerprintCacheInfo | None = None
    packed_fingerprints: np.ndarray | None = None
    row_bitcounts: np.ndarray | None = None
    if not spec.is_count_fingerprint:
        cache_info = build_synthon_fingerprint_cache(
            path,
            fingerprint_spec=spec,
            quiet_logs=quiet_logs,
            skip_bad_rows=skip_bad_rows,
            cache_prefix=cache_prefix,
            cache_dir=cache_dir,
            overwrite=overwrite_cache,
            chunk_size=cache_chunk_size,
            n_jobs=cache_n_jobs,
        )
        packed_fingerprints, row_bitcounts = _open_packed_fingerprint_cache(cache_info)

    by_reaction_slot: dict[tuple[str, int], list[SynthonRecord]] = {}
    row_lookup: dict[int, SynthonRecord] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        smiles_col = _find_column(reader.fieldnames, ["SMILES", "smiles"])
        id_col = _find_column(reader.fieldnames, ["synthon_id", "synton_id", "id"])
        slot_col = _find_column(reader.fieldnames, ["synthon#", "synton#", "slot", "position"])
        reaction_col = _find_column(reader.fieldnames, ["reaction_id", "rxn_id", "reaction"])

        kept = 0
        for row_number, row in enumerate(reader, start=2):
            smiles = str(row[smiles_col]).strip()
            synthon_id = str(row[id_col]).strip()
            slot_text = str(row[slot_col]).strip()
            reaction_id = str(row[reaction_col]).strip()
            if not smiles or not slot_text or not reaction_id:
                if skip_bad_rows:
                    continue
                raise ValueError(f"Missing SMILES/reaction/slot value at line {row_number}")
            try:
                slot = int(slot_text)
            except Exception as exc:
                if skip_bad_rows:
                    continue
                raise ValueError(f"Failed to parse row at line {row_number}: {row!r}") from exc
            fp: FingerprintValue | None = None
            if spec.is_count_fingerprint:
                try:
                    fp = fingerprint_smiles(smiles, spec)
                except Exception as exc:
                    if skip_bad_rows:
                        continue
                    raise ValueError(f"Failed to fingerprint row at line {row_number}: {row!r}") from exc
            elif skip_bad_rows:
                try:
                    fingerprint_smiles(smiles, spec)
                except Exception:
                    continue
            record = SynthonRecord(
                row_index=kept,
                smiles=smiles,
                synthon_id=synthon_id,
                slot=slot,
                fp=fp,
            )
            by_reaction_slot.setdefault((reaction_id, slot), []).append(record)
            row_lookup[kept] = record
            kept += 1

    reactions: list[ReactionIndex] = []
    reaction_ids = sorted({reaction_id for reaction_id, _slot in by_reaction_slot})
    for reaction_id in reaction_ids:
        slot_numbers = sorted(slot for rid, slot in by_reaction_slot if rid == reaction_id)
        slots = tuple(
            SlotIndex(slot=slot, records=tuple(by_reaction_slot[(reaction_id, slot)]))
            for slot in slot_numbers
        )
        if slots:
            reactions.append(ReactionIndex(reaction_id=reaction_id, slots=slots))

    return SynthonORIndex(
        reactions=tuple(reactions),
        fingerprint_spec=spec,
        row_lookup=row_lookup,
        packed_fingerprints=packed_fingerprints,
        row_bitcounts=row_bitcounts,
        fingerprint_cache=cache_info,
    )


def query_fingerprint_from_smiles(smiles: str, fingerprint_spec: FingerprintSpec) -> FingerprintValue:
    return fingerprint_smiles(smiles, fingerprint_spec)


def self_test_query_from_first_tuple(index: SynthonORIndex) -> FingerprintValue:
    for reaction in index.reactions:
        if all(slot.records for slot in reaction.slots):
            row_indices = [slot.records[0].row_index for slot in reaction.slots]
            return _union_row_fingerprints(row_indices, index)
    raise ValueError("No complete reaction tuple found in index")


def _top_indices(scores: Sequence[float], k: int) -> list[int]:
    if k <= 0 or len(scores) == 0:
        return []
    limit = min(int(k), len(scores))
    return sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx))[:limit]


def _selected_reactions(
    query_fp: FingerprintValue,
    index: SynthonORIndex,
    config: SearchConfig,
) -> list[tuple[str, tuple[SlotIndex, ...]]]:
    if index.uses_packed_fingerprints:
        if index.packed_fingerprints is None or index.row_bitcounts is None:
            raise ValueError("Packed fingerprint storage is unavailable")
        global_scores = _bulk_tanimoto_similarity_packed(
            _bitvect_to_packed_array(query_fp),
            index.packed_fingerprints,
            index.row_bitcounts,
        )
    else:
        global_scores = bulk_tanimoto_similarity(
            query_fp,
            tuple(index.row_lookup[row_idx].fp for row_idx in sorted(index.row_lookup)),
            index.fingerprint_spec,
        )
    ranked: list[tuple[float, str, tuple[SlotIndex, ...]]] = []
    for reaction in index.reactions:
        prepared_slots: list[SlotIndex] = []
        route_score = 0.0
        for slot in reaction.slots:
            local_scores = [float(global_scores[record.row_index]) for record in slot.records]
            keep = _top_indices(local_scores, config.candidate_bank_size)
            if not keep:
                prepared_slots = []
                break
            prepared_slot = SlotIndex(slot=slot.slot, records=tuple(slot.records[idx] for idx in keep))
            top_score = float(local_scores[keep[0]])
            prepared_slots.append(prepared_slot)
            route_score += top_score
        if prepared_slots:
            ranked.append((route_score, reaction.reaction_id, tuple(prepared_slots)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    if config.reaction_cap is None:
        return [(reaction_id, slots) for _score, reaction_id, slots in ranked]
    limit = max(0, int(config.reaction_cap))
    return [(reaction_id, slots) for _score, reaction_id, slots in ranked[:limit]]


def _pool_product_size(slots: Sequence[SlotIndex]) -> int:
    total = 1
    for slot in slots:
        total *= max(slot.size, 1)
    return total


def _trim_slots_to_exhaustive_limit(slots: tuple[SlotIndex, ...], limit: int) -> tuple[SlotIndex, ...]:
    if not slots:
        return tuple()
    active_limit = max(1, int(limit))
    if _pool_product_size(slots) <= active_limit:
        return slots

    arity = len(slots)
    base_cap = max(1, int(math.floor(active_limit ** (1.0 / max(arity, 1)))))
    sizes = [max(1, min(slot.size, base_cap)) for slot in slots]
    while math.prod(sizes) > active_limit:
        largest = max(range(len(sizes)), key=lambda idx: sizes[idx])
        if sizes[largest] <= 1:
            break
        sizes[largest] -= 1
    return tuple(slot.take(size) for slot, size in zip(slots, sizes, strict=True))


def materialize_candidates(
    query_fp: FingerprintValue,
    index: SynthonORIndex,
    *,
    config: SearchConfig | None = None,
) -> list[MaterializedCandidate]:
    active = config or SearchConfig()
    candidates: list[MaterializedCandidate] = []
    query_packed = _bitvect_to_packed_array(query_fp) if index.uses_packed_fingerprints else None
    query_on_bits = _bitcount_from_packed_array(query_packed) if query_packed is not None else None
    for reaction_id, slots in _selected_reactions(query_fp, index, active):
        trimmed_slots = _trim_slots_to_exhaustive_limit(slots, active.exhaustive_tuple_limit)
        if not trimmed_slots:
            continue
        for combo in product(*(slot.records for slot in trimmed_slots)):
            if index.uses_packed_fingerprints:
                if index.packed_fingerprints is None or query_packed is None or query_on_bits is None:
                    raise ValueError("Packed fingerprint storage is unavailable")
                combined = np.zeros((int(index.packed_fingerprints.shape[1]),), dtype=np.uint8)
                for record in combo:
                    np.bitwise_or(
                        combined,
                        np.asarray(index.packed_fingerprints[int(record.row_index)], dtype=np.uint8),
                        out=combined,
                    )
                approx_score = _packed_tanimoto_single(query_packed, query_on_bits, combined)
            else:
                fps = []
                for record in combo:
                    if record.fp is None:
                        raise ValueError("Missing in-memory fingerprint for row")
                    fps.append(record.fp)
                combined = union_fingerprints(fps, index.fingerprint_spec)
                approx_score = float(tanimoto_similarity(query_fp, combined, index.fingerprint_spec))
            candidates.append(
                MaterializedCandidate(
                    approx_score=float(approx_score),
                    reaction_id=reaction_id,
                    row_indices=tuple(int(record.row_index) for record in combo),
                )
            )
    candidates.sort(key=lambda item: (-item.approx_score, item.reaction_id, item.row_indices))
    return candidates


def candidate_to_hit(
    candidate: MaterializedCandidate,
    *,
    rank: int,
    index: SynthonORIndex,
) -> ApproxHit:
    records = tuple(index.row_lookup[row_idx] for row_idx in candidate.row_indices)
    return ApproxHit(
        rank=int(rank),
        reaction_id=str(candidate.reaction_id),
        approx_score=float(candidate.approx_score),
        synthon_ids=tuple(record.synthon_id for record in records),
        slots=tuple(record.slot for record in records),
        smiles=tuple(record.smiles for record in records),
        row_indices=tuple(record.row_index for record in records),
    )


def hits_from_candidates(
    candidates: Sequence[MaterializedCandidate],
    index: SynthonORIndex,
    *,
    start_rank: int = 1,
) -> list[ApproxHit]:
    return [
        candidate_to_hit(candidate, rank=rank, index=index)
        for rank, candidate in enumerate(candidates, start=max(1, int(start_rank)))
    ]


def _resolve_search_config(
    config: SearchConfig | None,
    *,
    preset: SearchPreset,
    top_n: int | None,
) -> SearchConfig:
    active = config or SearchConfig.preset(preset)
    if top_n is None:
        return active
    return replace(active, topk_products=max(0, int(top_n)))


def search_fingerprint(
    query_fp: FingerprintValue,
    index: SynthonORIndex,
    *,
    top_n: int | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    preset: SearchPreset = "accurate",
    config: SearchConfig | None = None,
) -> list[ApproxHit]:
    if min_score is not None and max_score is not None and float(min_score) > float(max_score):
        raise ValueError("min_score must be <= max_score")

    active = _resolve_search_config(config, preset=preset, top_n=top_n)
    candidates = materialize_candidates(query_fp, index, config=active)
    filtered = [
        candidate
        for candidate in candidates
        if (min_score is None or candidate.approx_score >= float(min_score))
        and (max_score is None or candidate.approx_score <= float(max_score))
    ]
    if top_n is not None:
        filtered = filtered[: max(0, int(top_n))]
    elif min_score is None and max_score is None:
        filtered = filtered[: max(0, int(active.topk_products))]
    return hits_from_candidates(filtered, index, start_rank=1)


def search_smiles(
    query_smiles: str,
    index: SynthonORIndex,
    *,
    top_n: int | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    preset: SearchPreset = "accurate",
    config: SearchConfig | None = None,
) -> list[ApproxHit]:
    query_fp = query_fingerprint_from_smiles(query_smiles, index.fingerprint_spec)
    return search_fingerprint(
        query_fp,
        index,
        top_n=top_n,
        min_score=min_score,
        max_score=max_score,
        preset=preset,
        config=config,
    )


def write_hits_jsonl(hits: Sequence[ApproxHit], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for hit in hits:
            handle.write(json.dumps(hit.to_json_dict(), sort_keys=True) + "\n")
