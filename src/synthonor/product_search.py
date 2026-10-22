"""Assembled-product search backed by the private shared identity engine."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from rdkit import Chem
from synthonor.synthon_or_rdkit import ApproxHit, SearchConfig, search_smiles
from synthonor.fingerprints import (
    fingerprint_smiles,
    tanimoto_similarity,
    union_fingerprints,
)


@dataclass(frozen=True)
class ProductHit:
    hit: ApproxHit
    smiles: str
    score: float


class ProductSearchEngine:
    """Build once per loaded space, then share across queries.

    ``assemble(route, row_indices)`` must use the space's real forward assembler.
    Rows refer to the input SynthonORIndex. For reaction-based assembly supply
    the exact forward RDKit templates in ``reference_reactions``. The shared
    inverse algorithm stays in dmc-synthon-core; no product enumeration is built.
    """

    def __init__(self, index, assemble, *, cache_dir=None, reference_reactions=None):
        # Keep ordinary SynthonOR imports usable without private dependencies.
        try:
            from dmc_synthon_core.identity import build_identity_index
            from dmc_synthon_core.runtime import AssemblyResult, Candidate
        except ModuleNotFoundError as exc:
            if exc.name != "dmc_synthon_core":
                raise
            raise ImportError(
                "ProductSearchEngine requires the private dmc-synthon-core "
                "integration (validated with 0.4.6; Python 3.11 or 3.12). Install it from "
                "your authorized private checkout."
            ) from exc

        self.index = index
        self.assemble = assemble
        original_rows = tuple(sorted(index.row_lookup))
        dense = {row: i for i, row in enumerate(original_rows)}
        self.original_rows = original_rows
        runtime = SimpleNamespace(
            synthon_ids=np.asarray([str(row) for row in original_rows]),
            synthon_smiles=np.asarray(
                [index.row_lookup[row].smiles for row in original_rows]
            ),
            reactions=[r.reaction_id for r in index.reactions],
            slot_tables={
                r.reaction_id: [
                    np.asarray([dense[x.row_index] for x in s.records], dtype=np.int64)
                    for s in r.slots
                ]
                for r in index.reactions
            },
        )
        runtime.make_candidate = lambda route, ids: Candidate(
            route + ":" + ",".join(ids), route, ids
        )

        def assemble_many(candidates):
            results = []
            for candidate in candidates:
                smiles = assemble(
                    candidate.reaction_id,
                    tuple(int(row) for row in candidate.synthon_ids),
                )
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                results.append(
                    AssemblyResult(Chem.MolToSmiles(mol) if mol else None, mol)
                )
            return results

        runtime.assemble_many = assemble_many
        if reference_reactions is not None:
            runtime.reference_reactions = reference_reactions
        self.identity_index = build_identity_index(runtime, cache_dir=cache_dir)

    def search(
        self,
        query_smiles,
        *,
        top_n=100,
        min_score=None,
        max_score=None,
        preset="fast",
        config: SearchConfig | None = None,
    ):
        if top_n <= 0:
            return []
        if min_score is not None and max_score is not None and min_score > max_score:
            raise ValueError("min_score must be <= max_score")
        molecule = Chem.MolFromSmiles(query_smiles)
        if molecule is None:
            raise ValueError("invalid SMILES")
        canonical = Chem.MolToSmiles(molecule)
        spec = self.index.fingerprint_spec
        query_fp = fingerprint_smiles(canonical, spec)
        hits = search_smiles(
            canonical, self.index, top_n=top_n, preset=preset, config=config
        )
        identity = self.identity_index.find(canonical).match
        verified = {}
        if identity is not None:
            rows = tuple(self.original_rows[row] for row in identity.rows)
            records = [self.index.row_lookup[row] for row in rows]
            union = union_fingerprints(
                [fingerprint_smiles(r.smiles, spec) for r in records], spec
            )
            exact = ApproxHit(
                rank=1,
                reaction_id=identity.route,
                approx_score=float(tanimoto_similarity(query_fp, union, spec)),
                synthon_ids=tuple(r.synthon_id for r in records),
                slots=tuple(r.slot for r in records),
                smiles=tuple(r.smiles for r in records),
                row_indices=rows,
            )
            key = (identity.route, rows)
            hits = (
                [exact] + [h for h in hits if (h.reaction_id, h.row_indices) != key]
            )[:top_n]
            verified[key] = identity.assembly.smiles
        products = []
        seen = set()
        for hit in hits:
            key = (hit.reaction_id, hit.row_indices)
            smiles = verified.get(key) or self.assemble(*key)
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                continue
            smiles = Chem.MolToSmiles(mol)
            if smiles in seen:
                continue
            seen.add(smiles)
            score = float(
                tanimoto_similarity(query_fp, fingerprint_smiles(smiles, spec), spec)
            )
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue
            products.append(ProductHit(hit, smiles, score))
        products.sort(key=lambda p: (-p.score, p.smiles != canonical))
        return products
