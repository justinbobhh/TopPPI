from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

PIPELINE_VERSION = "2.0.0"
WINDOW_SIZE = 5
WINDOW_STRIDE = 5

AA3_TO_1: Mapping[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",

    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
    "UNK": "X", "HSD": "H", "HSE": "H", "HSP": "H", "SEP": "S",
    "TPO": "T", "PTR": "Y", "CSO": "C", "CME": "C", "KCX": "K",
    "MLY": "K", "FME": "M", "PCA": "E", "LLP": "K",
}

@dataclass(frozen=True)
class AtomRecord:
    name: str
    element: str
    coord: np.ndarray
    occupancy: float
    altloc: str

@dataclass(frozen=True)
class ResidueRecord:
    chain_id: str
    resseq: int
    insertion_code: str
    resname: str
    amino_acid: str
    atoms: Tuple[AtomRecord, ...]

    @property
    def key(self) -> str:
        icode = self.insertion_code if self.insertion_code else "."
        return f"{self.chain_id}:{self.resseq}:{icode}"

    @property
    def heavy_coords(self) -> np.ndarray:
        coords = [a.coord for a in self.atoms if a.element.upper() not in {"H", "D"}]
        if not coords:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack(coords).astype(np.float64, copy=False)

    @property
    def ca_coord(self) -> np.ndarray | None:
        for atom in self.atoms:
            if atom.name == "CA":
                return atom.coord.astype(np.float64, copy=False)
        return None

def _infer_element(atom_name: str) -> str:
    stripped = "".join(ch for ch in atom_name if ch.isalpha()).upper()
    if not stripped:
        return ""

    return stripped[0]

def parse_pdb_first_model(pdb_path: Path) -> Dict[str, List[ResidueRecord]]:

    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    raw: Dict[str, Dict[Tuple[int, str, str], Dict[str, AtomRecord]]] = {}
    seen_model = False
    in_first_model = True

    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[0:6].strip().upper()
            if record == "MODEL":
                if seen_model:
                    in_first_model = False
                else:
                    seen_model = True
                    in_first_model = True
                continue
            if record == "ENDMDL" and in_first_model:
                break
            if not in_first_model or record not in {"ATOM", "HETATM"}:
                continue
            if len(line) < 54:
                continue

            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_1:
                continue
            chain_id = line[21:22].strip() or "_"
            try:
                resseq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            insertion_code = line[26:27].strip()
            atom_name = line[12:16].strip().upper()
            altloc = line[16:17].strip()
            try:
                occupancy = float(line[54:60].strip() or "0")
            except ValueError:
                occupancy = 0.0
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            element = element or _infer_element(atom_name)

            atom = AtomRecord(
                name=atom_name,
                element=element,
                coord=np.asarray([x, y, z], dtype=np.float64),
                occupancy=occupancy,
                altloc=altloc,
            )
            residue_key = (resseq, insertion_code, resname)
            atom_map = raw.setdefault(chain_id, {}).setdefault(residue_key, {})
            prior = atom_map.get(atom_name)

            rank = (altloc not in {"", "A", "1"}, -occupancy, altloc)
            if prior is None:
                atom_map[atom_name] = atom
            else:
                prior_rank = (
                    prior.altloc not in {"", "A", "1"},
                    -prior.occupancy,
                    prior.altloc,
                )
                if rank < prior_rank:
                    atom_map[atom_name] = atom

    parsed: Dict[str, List[ResidueRecord]] = {}
    for chain_id, residue_map in raw.items():
        residues: List[ResidueRecord] = []
        for (resseq, icode, resname), atom_map in residue_map.items():
            atoms = tuple(atom_map[name] for name in sorted(atom_map))
            if not atoms:
                continue
            residues.append(
                ResidueRecord(
                    chain_id=chain_id,
                    resseq=resseq,
                    insertion_code=icode,
                    resname=resname,
                    amino_acid=AA3_TO_1[resname],
                    atoms=atoms,
                )
            )
        parsed[chain_id] = residues
    return parsed

def load_chain(pdb_path: Path, chain_id: str) -> List[ResidueRecord]:
    chains = parse_pdb_first_model(Path(pdb_path))
    if chain_id not in chains:
        available = ", ".join(sorted(chains)) or "none"
        raise ValueError(
            f"Chain {chain_id!r} not found in {pdb_path}; available chains: {available}"
        )
    residues = chains[chain_id]
    if not residues:
        raise ValueError(f"Chain {chain_id!r} has no recognised amino-acid residues")
    return residues

def chain_sequence(residues: Sequence[ResidueRecord]) -> str:
    return "".join(res.amino_acid for res in residues)

def read_complex_manifest(path: Path) -> List[dict]:
    required = {"panel_order", "pdb_id", "pdb_file", "chain_a", "chain_b"}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Complex manifest is empty: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    seen = set()
    for row in rows:
        row["pdb_id"] = row["pdb_id"].strip().upper()
        row["chain_a"] = row["chain_a"].strip()
        row["chain_b"] = row["chain_b"].strip()
        row["pdb_file"] = row["pdb_file"].strip()
        try:
            row["panel_order"] = int(row["panel_order"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid panel_order for {row.get('pdb_id')}") from exc
        if row["pdb_id"] in seen:
            raise ValueError(f"Duplicate PDB ID in manifest: {row['pdb_id']}")
        if row["chain_a"] == row["chain_b"]:
            raise ValueError(f"Identical chains selected for {row['pdb_id']}")
        seen.add(row["pdb_id"])
    rows.sort(key=lambda x: x["panel_order"])
    return rows

def resolve_pdb_path(row: Mapping[str, object], pdb_dir: Path) -> Path:
    candidate = Path(str(row["pdb_file"]))
    return candidate if candidate.is_absolute() else Path(pdb_dir) / candidate

def atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

def atomic_write_csv(dataframe, path: Path) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        dataframe.to_csv(tmp_name, index=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def file_identity(path: Path, include_sha256: bool = False) -> dict:
    stat = Path(path).stat()
    result = {
        "path": str(Path(path).resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = sha256_file(Path(path))
    return result

def stable_fingerprint(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def finite_or_none(value: float | int | np.number | None):
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None
