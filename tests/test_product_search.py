from dataclasses import replace

import pytest
from rdkit import Chem
from synthonor import ApproxHit, FingerprintSpec
from synthonor.fingerprints import fingerprint_smiles
from synthonor.synthon_or_rdkit import (
    ReactionIndex,
    SlotIndex,
    SynthonORIndex,
    SynthonRecord,
)

from synthonor.product_search import ProductSearchEngine


def fixture_engine(tmp_path, monkeypatch):
    pytest.importorskip("dmc_synthon_core.identity")
    # Sparse original row IDs exercise the adapter's dense row mapping.
    smiles = {5: "C[C@H](O)C[*:1]", 10: "C[C@@H](O)C[*:1]", 20: "N[*:1]"}
    records = {
        row: SynthonRecord(
            row,
            smi,
            str(row),
            0 if row < 20 else 1,
            fp=fingerprint_smiles(smi, FingerprintSpec()),
        )
        for row, smi in smiles.items()
    }
    index = SynthonORIndex(
        reactions=(
            ReactionIndex(
                "r",
                (SlotIndex(0, (records[5], records[10])), SlotIndex(1, (records[20],))),
            ),
        ),
        fingerprint_spec=FingerprintSpec(),
        row_lookup=records,
    )
    calls = []

    def assemble(route, rows):
        calls.append((route, rows))
        return Chem.MolToSmiles(
            Chem.molzip(Chem.MolFromSmiles(".".join(smiles[r] for r in rows)))
        )

    query = assemble("r", (10, 20))
    wrong = ApproxHit(
        1, "r", 1.0, ("5", "20"), (0, 1), (smiles[5], smiles[20]), (5, 20)
    )
    monkeypatch.setattr(
        "synthonor.product_search.search_smiles",
        lambda *args, **kwargs: [wrong],
    )
    engine = ProductSearchEngine(index, assemble, cache_dir=tmp_path)
    return engine, query, wrong, calls


def test_pruned_identity_replaces_one_candidate_and_reuses_assembly(
    tmp_path, monkeypatch
):
    engine, query, _, calls = fixture_engine(tmp_path, monkeypatch)
    calls.clear()
    products = engine.search(query, top_n=1)
    assert len(products) == 1
    assert products[0].smiles == query
    assert products[0].hit.row_indices == (10, 20)
    assert products[0].score == 1.0
    assert calls.count(("r", (10, 20))) == 1
    assert products[0].hit.approx_score < 1.0
    assert engine.search(query, top_n=1, max_score=0.99) == []


def test_full_product_ties_prefer_isomeric_identity(tmp_path, monkeypatch):
    engine, query, wrong, _ = fixture_engine(tmp_path, monkeypatch)
    exact = replace(wrong, rank=2, synthon_ids=("10", "20"), row_indices=(10, 20))
    monkeypatch.setattr(
        "synthonor.product_search.search_smiles",
        lambda *args, **kwargs: [wrong, exact],
    )
    # Test ranking independently of the inverse-proposal insertion order.
    monkeypatch.setattr(
        engine.identity_index,
        "find",
        lambda query: type("Result", (), {"match": None})(),
    )
    products = engine.search(query, top_n=2)
    assert [p.score for p in products] == [1.0, 1.0]
    assert products[0].smiles == query


def test_invalid_and_absent_queries(tmp_path, monkeypatch):
    engine, _, _, _ = fixture_engine(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="invalid SMILES"):
        engine.search("invalid")
    assert all(p.smiles != "CCCCCCCC" for p in engine.search("CCCCCCCC"))


def test_real_candidate_search_and_full_product_threshold(tmp_path):
    pytest.importorskip("dmc_synthon_core.identity")
    from synthonor import load_synthon_or_index

    table = tmp_path / "synthons.tsv"
    table.write_text(
        "smiles\tsynthon_id\tposition\treaction_id\n"
        "C[C@H](O)C[*:1]\ta\t1\tr\n"
        "C[C@@H](O)C[*:1]\tb\t1\tr\n"
        "N[*:1]\tc\t2\tr\n"
    )
    index = load_synthon_or_index(table)

    def assemble(route, rows):
        assert route == "r"
        return Chem.MolToSmiles(
            Chem.molzip(
                Chem.MolFromSmiles(
                    ".".join(index.row_lookup[row].smiles for row in rows)
                )
            )
        )

    query = assemble("r", (1, 2))
    engine = ProductSearchEngine(index, assemble, cache_dir=tmp_path / "identity")
    products = engine.search(query, top_n=1, min_score=1.0)
    assert len(products) == 1
    assert products[0].smiles == query
    assert products[0].score == 1.0
    assert products[0].hit.approx_score < 1.0


@pytest.mark.parametrize(
    "template",
    [
        "[C:1]-[#92].[N:2]-[#92]>>[C:1]-C(=O)-[N:2]",
        "[C:1]-[#92].[N:2]-[#92]>>[C:1]-C1=N-C=C(-[N:2])-S1",
    ],
)
def test_reference_template_retrieval(template, monkeypatch):
    pytest.importorskip("dmc_synthon_core.identity")
    from rdkit.Chem import rdChemReactions

    reaction = rdChemReactions.ReactionFromSmarts(template)
    records = {
        11: SynthonRecord(11, "CC[U]", "a", 0),
        21: SynthonRecord(21, "CN[U]", "b", 1),
    }
    index = SynthonORIndex(
        reactions=(
            ReactionIndex(
                "r",
                (
                    SlotIndex(0, (records[11],)),
                    SlotIndex(1, (records[21],)),
                ),
            ),
        ),
        fingerprint_spec=FingerprintSpec(),
        row_lookup=records,
    )

    def assemble(route, rows):
        assert route == "r"
        product = reaction.RunReactants(
            tuple(Chem.MolFromSmiles(records[row].smiles) for row in rows)
        )[0][0]
        Chem.SanitizeMol(product)
        return Chem.MolToSmiles(product)

    # An empty approximate shortlist still allows verified inverse recovery.
    monkeypatch.setattr("synthonor.product_search.search_smiles", lambda *a, **k: [])
    query = assemble("r", (11, 21))
    engine = ProductSearchEngine(index, assemble, reference_reactions={"r": reaction})
    products = engine.search(query, top_n=1)
    assert len(products) == 1
    assert products[0].smiles == query
    assert products[0].hit.row_indices == (11, 21)
    assert engine.search("CCCC(=O)NC", top_n=1) == []


def test_import_without_private_core_and_actionable_install_error():
    import subprocess
    import sys

    # Also proves importing the public package never attempts a private import.
    program = """
import importlib.abc
import sys
class BlockPrivateCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dmc_synthon_core" or fullname.startswith("dmc_synthon_core."):
            raise ModuleNotFoundError("private core intentionally absent", name=fullname)
sys.meta_path.insert(0, BlockPrivateCore())
from synthonor import ProductSearchEngine, search_smiles
assert callable(search_smiles)
try:
    ProductSearchEngine(None, None)
except ImportError as exc:
    assert "authorized private checkout" in str(exc)
else:
    raise AssertionError("missing core was silently ignored")
"""
    subprocess.run([sys.executable, "-c", program], check=True)
