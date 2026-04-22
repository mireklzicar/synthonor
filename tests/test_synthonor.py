from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from synthonor import (
    FingerprintSpec,
    SearchConfig,
    __version__,
    build_synthon_fingerprint_cache,
    example_space_path,
    load_synthon_or_index,
    search_fingerprint,
    search_smiles,
)
from synthonor.cli import main as cli_main
from synthonor.fingerprints import fingerprint_smiles


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_search_smoke_and_self_recovery(tmp_path: Path) -> None:
    syntons = _write(
        tmp_path / "syntons.txt",
        "\n".join(
            [
                "SMILES\tsynton_id\tsynton#\treaction_id",
                "C[U]\ta1\t1\tr1",
                "CC[U]\tb1\t2\tr1",
                "N[U]\ta2\t1\tr1",
                "O[U]\tb2\t2\tr1",
            ]
        )
        + "\n",
    )
    index = load_synthon_or_index(syntons, fingerprint_spec=FingerprintSpec(kind="ecfp4"))
    hits = search_smiles(
        "CCC",
        index,
        config=SearchConfig(
            reaction_cap=1,
            candidate_bank_size=2,
            topk_products=4,
        ),
    )
    assert hits
    assert hits[0].reaction_id == "r1"
    assert hits[0].synthon_ids == ("a1", "b1")


def test_search_presets_and_thresholds(tmp_path: Path) -> None:
    syntons = _write(
        tmp_path / "syntons.txt",
        "\n".join(
            [
                "SMILES\tsynton_id\tsynton#\treaction_id",
                "C[U]\ta1\t1\tr1",
                "N[U]\ta2\t1\tr1",
                "CC[U]\tb1\t2\tr1",
                "C=C[U]\tb2\t2\tr1",
            ]
        )
        + "\n",
    )
    index = load_synthon_or_index(syntons, fingerprint_spec=FingerprintSpec(kind="ecfp4"))

    top_hits = search_smiles("CCN", index, top_n=2, preset="accurate")
    assert len(top_hits) == 2
    assert top_hits[0].synthon_ids == ("a2", "b1")

    threshold_hits = search_smiles("CCN", index, min_score=0.3, preset="accurate")
    assert threshold_hits
    assert all(hit.approx_score >= 0.3 for hit in threshold_hits)

    accurate = SearchConfig.preset("accurate", topk_products=7)
    very_accurate = SearchConfig.preset("very_accurate", topk_products=7)
    fast = SearchConfig.preset("fast", topk_products=7)
    assert accurate.reaction_cap == 8
    assert fast.reaction_cap == 1
    assert very_accurate.reaction_cap is None


def test_bit_fingerprint_cache_is_built_and_reused(tmp_path: Path) -> None:
    syntons = _write(
        tmp_path / "syntons.txt",
        "\n".join(
            [
                "SMILES\tsynton_id\tsynton#\treaction_id",
                "C[U]\ta1\t1\tr1",
                "N[U]\ta2\t1\tr1",
                "CC[U]\tb1\t2\tr1",
                "O[U]\tb2\t2\tr1",
            ]
        )
        + "\n",
    )
    spec = FingerprintSpec(kind="ecfp4")
    cache_info = build_synthon_fingerprint_cache(syntons, fingerprint_spec=spec, chunk_size=2, n_jobs=1)
    meta = json.loads(Path(cache_info.meta_path).read_text(encoding="utf-8"))
    assert cache_info.num_rows == 4
    assert meta["fingerprint"] == "ecfp4"
    assert Path(cache_info.packed_fps_path).exists()
    assert Path(cache_info.bitcounts_path).exists()

    first_index = load_synthon_or_index(syntons, fingerprint_spec=spec, cache_chunk_size=2, cache_n_jobs=1)
    second_index = load_synthon_or_index(syntons, fingerprint_spec=spec, cache_chunk_size=2, cache_n_jobs=1)
    assert first_index.uses_packed_fingerprints
    assert second_index.uses_packed_fingerprints
    assert first_index.fingerprint_cache is not None
    assert second_index.fingerprint_cache is not None
    assert first_index.fingerprint_cache.cache_prefix == second_index.fingerprint_cache.cache_prefix

    hits = search_smiles("CCN", second_index, top_n=2, preset="accurate")
    assert len(hits) == 2
    assert hits[0].synthon_ids == ("a2", "b1")


def test_search_fingerprint_mode(tmp_path: Path) -> None:
    syntons = _write(
        tmp_path / "syntons.txt",
        "\n".join(
            [
                "SMILES\tsynton_id\tsynton#\treaction_id",
                "c1ccccc1[U]\ta1\t1\tr1",
                "N[U]\ta2\t1\tr1",
                "CC[U]\tb1\t2\tr1",
                "O[U]\tb2\t2\tr1",
            ]
        )
        + "\n",
    )
    spec = FingerprintSpec(kind="morgan_counts", union_mode="sum")
    index = load_synthon_or_index(syntons, fingerprint_spec=spec)
    query_fp = fingerprint_smiles("CCN", index.fingerprint_spec)
    hits = search_fingerprint(query_fp, index, top_n=2, preset="accurate")
    assert len(hits) == 2
    assert hits[0].synthon_ids == ("a2", "b1")


def test_cache_invalidation_tracks_mtime_and_hash(tmp_path: Path) -> None:
    syntons = _write(
        tmp_path / "syntons.txt",
        "\n".join(
            [
                "SMILES\tsynton_id\tsynton#\treaction_id",
                "C[U]\ta1\t1\tr1",
                "CC[U]\tb1\t2\tr1",
            ]
        )
        + "\n",
    )
    spec = FingerprintSpec(kind="ecfp4")
    first_cache = build_synthon_fingerprint_cache(syntons, fingerprint_spec=spec)
    first_meta = json.loads(Path(first_cache.meta_path).read_text(encoding="utf-8"))

    time.sleep(0.01)
    _write(
        syntons,
        "\n".join(
            [
                "SMILES\tsynton_id\tsynton#\treaction_id",
                "N[U]\ta1\t1\tr1",
                "CC[U]\tb1\t2\tr1",
            ]
        )
        + "\n",
    )

    second_cache = build_synthon_fingerprint_cache(syntons, fingerprint_spec=spec)
    second_meta = json.loads(Path(second_cache.meta_path).read_text(encoding="utf-8"))

    assert first_meta["version"] == second_meta["version"] == 2
    assert first_meta["source"]["size"] == second_meta["source"]["size"]
    assert first_meta["source"]["mtime_ns"] != second_meta["source"]["mtime_ns"]
    assert first_meta["source"]["content_sha256"] != second_meta["source"]["content_sha256"]


def test_example_space_path_is_available() -> None:
    with example_space_path() as path:
        assert path.exists()
        with path.open("r", encoding="utf-8") as handle:
            header = [value.strip().lower() for value in handle.readline().strip().split("\t")]
    assert "smiles" in header
    assert "reaction_id" in header


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert int(exc.value.code) == 0
    out = capsys.readouterr().out.strip()
    assert out == f"synthonor {__version__}"


def test_cli_requires_explicit_query_or_test(tmp_path: Path) -> None:
    syntons = _write(
        tmp_path / "syntons.txt",
        "\n".join(
            [
                "SMILES\tsynton_id\tsynton#\treaction_id",
                "C[U]\ta1\t1\tr1",
                "CC[U]\tb1\t2\tr1",
            ]
        )
        + "\n",
    )
    with pytest.raises(SystemExit) as missing_mode:
        cli_main([str(syntons)])
    assert int(missing_mode.value.code) == 2

    with pytest.raises(SystemExit) as conflicting_mode:
        cli_main([str(syntons), "--query", "CCN", "--test"])
    assert int(conflicting_mode.value.code) == 2


def test_cli_test_mode_smoke(tmp_path: Path) -> None:
    syntons = _write(
        tmp_path / "syntons.txt",
        "\n".join(
            [
                "SMILES\tsynton_id\tsynton#\treaction_id",
                "C[U]\ta1\t1\tr1",
                "N[U]\ta2\t1\tr1",
                "CC[U]\tb1\t2\tr1",
                "O[U]\tb2\t2\tr1",
            ]
        )
        + "\n",
    )
    output = tmp_path / "hits.jsonl"
    code = cli_main([str(syntons), "--test", "--top-n", "1", "--output", str(output), "--preset", "fast"])
    assert int(code) == 0
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
