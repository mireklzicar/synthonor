from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


CONNECTOR_ATOMIC_NUMS = (92, 93)

FingerprintKind = Literal[
    "ecfp4",
    "ecfp6",
    "morgan",
    "morgan_counts",
    "rdkit",
    "patternfp",
    "atom_pair",
    "topological_torsion",
]
ConnectorPolicy = Literal["keep"]
UnionMode = Literal["or", "sum", "max"]
FingerprintValue = DataStructs.ExplicitBitVect | dict[int, int]


@dataclass(frozen=True)
class FingerprintSpec:
    kind: FingerprintKind = "ecfp4"
    fp_size: int = 2048
    radius: int = 2
    include_chirality: bool = False
    use_features: bool = False
    union_mode: UnionMode = "or"
    connector_policy: ConnectorPolicy = "keep"

    def normalized(self) -> "FingerprintSpec":
        kind = str(self.kind).strip().lower()
        if str(self.connector_policy).strip().lower() != "keep":
            raise ValueError("synthonor only supports connector_policy='keep'")
        if kind == "ecfp4":
            return replace(self, kind="morgan", radius=2, union_mode="or")
        if kind == "ecfp6":
            return replace(self, kind="morgan", radius=3, union_mode="or")
        if kind == "morgan_counts" and str(self.union_mode).lower() not in {"sum", "max"}:
            return replace(self, union_mode="sum")
        if kind != "morgan_counts":
            return replace(self, union_mode="or")
        return self

    @property
    def is_count_fingerprint(self) -> bool:
        return str(self.normalized().kind) == "morgan_counts"

    @property
    def label(self) -> str:
        spec = self.normalized()
        if spec.kind == "morgan" and spec.radius == 2:
            return "ecfp4"
        if spec.kind == "morgan" and spec.radius == 3:
            return "ecfp6"
        if spec.kind == "morgan_counts":
            return f"morgan_counts_{spec.union_mode}"
        return str(spec.kind)


def quiet_rdkit_logs(enabled: bool = True) -> None:
    if enabled:
        RDLogger.DisableLog("rdApp.*")


def mol_from_smiles(smiles: str, *, connector_policy: ConnectorPolicy = "keep") -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    if str(connector_policy).strip().lower() != "keep":
        raise ValueError("synthonor only supports connector_policy='keep'")
    return mol


def canonical_smiles(smiles: str, *, connector_policy: ConnectorPolicy = "keep") -> str:
    mol = mol_from_smiles(smiles, connector_policy=connector_policy)
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _morgan_generator(spec: FingerprintSpec):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=int(spec.radius),
        fpSize=int(spec.fp_size),
        includeChirality=bool(spec.include_chirality),
        atomInvariantsGenerator=(
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            if bool(spec.use_features)
            else None
        ),
    )


def _rdkit_generator(spec: FingerprintSpec):
    return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=int(spec.fp_size))


def _atom_pair_generator(spec: FingerprintSpec):
    return rdFingerprintGenerator.GetAtomPairGenerator(
        fpSize=int(spec.fp_size),
        includeChirality=bool(spec.include_chirality),
    )


def _torsion_generator(spec: FingerprintSpec):
    return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=int(spec.fp_size),
        includeChirality=bool(spec.include_chirality),
    )


def fingerprint_mol(mol: Chem.Mol, spec: FingerprintSpec) -> FingerprintValue:
    spec = spec.normalized()
    if spec.kind == "morgan":
        return _morgan_generator(spec).GetFingerprint(mol)
    if spec.kind == "morgan_counts":
        fp = rdMolDescriptors.GetHashedMorganFingerprint(
            mol,
            int(spec.radius),
            nBits=int(spec.fp_size),
            useFeatures=bool(spec.use_features),
            useChirality=bool(spec.include_chirality),
        )
        return {int(bit_id): int(count) for bit_id, count in fp.GetNonzeroElements().items()}
    if spec.kind == "rdkit":
        return _rdkit_generator(spec).GetFingerprint(mol)
    if spec.kind == "patternfp":
        return Chem.PatternFingerprint(mol, fpSize=int(spec.fp_size))
    if spec.kind == "atom_pair":
        return _atom_pair_generator(spec).GetFingerprint(mol)
    if spec.kind == "topological_torsion":
        return _torsion_generator(spec).GetFingerprint(mol)
    raise ValueError(f"Unsupported fingerprint kind: {spec.kind!r}")


def fingerprint_smiles(smiles: str, spec: FingerprintSpec) -> FingerprintValue:
    return fingerprint_mol(
        mol_from_smiles(smiles, connector_policy=spec.connector_policy),
        spec,
    )


def _count_tanimoto(left: dict[int, int], right: dict[int, int]) -> float:
    if not left and not right:
        return 1.0
    shared_keys = set(left) | set(right)
    numer = 0
    denom = 0
    for key in shared_keys:
        left_value = int(left.get(key, 0))
        right_value = int(right.get(key, 0))
        numer += min(left_value, right_value)
        denom += max(left_value, right_value)
    return float(numer / max(denom, 1))


def tanimoto_similarity(query_fp: FingerprintValue, target_fp: FingerprintValue, spec: FingerprintSpec) -> float:
    spec = spec.normalized()
    if spec.is_count_fingerprint:
        return _count_tanimoto(
            dict(query_fp),  # type: ignore[arg-type]
            dict(target_fp),  # type: ignore[arg-type]
        )
    return float(DataStructs.TanimotoSimilarity(query_fp, target_fp))  # type: ignore[arg-type]


def bulk_tanimoto_similarity(
    query_fp: FingerprintValue,
    target_fps: Sequence[FingerprintValue],
    spec: FingerprintSpec,
) -> list[float]:
    spec = spec.normalized()
    if spec.is_count_fingerprint:
        return [tanimoto_similarity(query_fp, target_fp, spec) for target_fp in target_fps]
    return [float(value) for value in DataStructs.BulkTanimotoSimilarity(query_fp, list(target_fps))]  # type: ignore[arg-type]


def union_fingerprints(fps: Sequence[FingerprintValue], spec: FingerprintSpec) -> FingerprintValue:
    spec = spec.normalized()
    if not fps:
        raise ValueError("Cannot combine an empty fingerprint sequence")
    if spec.is_count_fingerprint:
        union_mode = str(spec.union_mode).lower()
        combined: dict[int, int] = {}
        for fp in fps:
            current = dict(fp)  # type: ignore[arg-type]
            for bit_id, value in current.items():
                if union_mode == "sum":
                    combined[bit_id] = int(combined.get(bit_id, 0) + int(value))
                elif union_mode == "max":
                    combined[bit_id] = max(int(combined.get(bit_id, 0)), int(value))
                else:
                    raise ValueError(f"Unsupported count union mode: {spec.union_mode!r}")
        return combined
    first = fps[0]
    combined = DataStructs.ExplicitBitVect(first.GetNumBits())  # type: ignore[union-attr]
    for fp in fps:
        combined |= fp  # type: ignore[operator]
    return combined
