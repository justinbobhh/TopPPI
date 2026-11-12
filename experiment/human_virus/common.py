from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

VALID_FOLDS: Tuple[str, ...] = ("VF1", "VF2", "VF3", "VF4", "VF5")
VALID_SPLITS: Tuple[str, ...] = ("C1Virus", "C2Virus", "Test", "Train")
SPLIT_ALIASES: Mapping[str, str] = {
    "c1": "C1Virus",
    "c1virus": "C1Virus",
    "c2": "C2Virus",
    "c2virus": "C2Virus",
    "test": "Test",
    "train": "Train",
}
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "protein_a_id",
    "protein_b_id",
    "seq_a",
    "seq_b",
    "label",
)
PREDICTION_COLUMNS: Tuple[str, ...] = (
    "row_index",
    "protein_a_id",
    "protein_b_id",
    "label",
    "logit",
    "probability",
    "prediction",
)
METRIC_NAMES: Tuple[str, ...] = (
    "AUROC",
    "AUPRC",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "MCC",
)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def canonical_folds(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return list(VALID_FOLDS)
    result: List[str] = []
    for value in values:
        for token in value.split(","):
            fold = token.strip().upper()
            if fold not in VALID_FOLDS:
                raise ValueError(
                    f"Unsupported fold {token!r}; expected one of {', '.join(VALID_FOLDS)}"
                )
            if fold not in result:
                result.append(fold)
    return result

def canonical_splits(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return ["C1Virus", "C2Virus"]
    result: List[str] = []
    for value in values:
        for token in value.split(","):
            key = token.strip().lower()
            if key not in SPLIT_ALIASES:
                raise ValueError(
                    f"Unsupported split {token!r}; expected C1Virus, C2Virus, Test, or Train"
                )
            split = SPLIT_ALIASES[key]
            if split not in result:
                result.append(split)
    return result

def csv_path(data_root: Path, fold: str, split: str) -> Path:
    return data_root / fold / f"VirusAppear_{fold}_{split}.csv"

def checkpoint_path(checkpoint_dir: Path, fold: str) -> Path:
    return checkpoint_dir / f"best_model_{fold}.pt"

def quick_file_digest(path: Path, block_bytes: int = 1024 * 1024) -> str:

    size = path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        hasher.update(handle.read(block_bytes))
        if size > block_bytes:
            handle.seek(max(0, size - block_bytes))
            hasher.update(handle.read(block_bytes))
    return hasher.hexdigest()

def full_file_sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()

def fingerprint_file(path: Path, mode: str = "quick") -> Dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    result: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "fingerprint_mode": mode,
    }
    if mode == "quick":
        result["quick_sha256"] = quick_file_digest(path)
    elif mode == "full":
        result["sha256"] = full_file_sha256(path)
    elif mode != "none":
        raise ValueError("fingerprint mode must be one of: none, quick, full")
    return result

def audit_csv(
    path: Path,
    *,
    hash_mode: str = "quick",
    tokenizer_max_length: int = 800,
) -> Dict[str, Any]:

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    label_counts: Counter[str] = Counter()
    seen_pairs: Dict[Tuple[str, str], str] = {}
    duplicate_rows = 0
    conflicting_pair_labels = 0
    empty_ids = 0
    empty_sequences = 0
    malformed_rows = 0
    max_seq_a_length = 0
    max_seq_b_length = 0
    seq_a_truncated = 0
    seq_b_truncated = 0
    row_count = 0
    examples: List[Dict[str, Any]] = []

    raw_residue_limit = max(0, tokenizer_max_length - 2)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}; "
                f"found: {', '.join(fieldnames)}"
            )

        for row_index, row in enumerate(reader):
            row_count += 1
            if None in row:
                malformed_rows += 1

            protein_a_id = (row.get("protein_a_id") or "").strip()
            protein_b_id = (row.get("protein_b_id") or "").strip()
            seq_a = "".join((row.get("seq_a") or "").split())
            seq_b = "".join((row.get("seq_b") or "").split())
            label = (row.get("label") or "").strip()

            label_counts[label] += 1
            if not protein_a_id or not protein_b_id:
                empty_ids += 1
            if not seq_a or not seq_b:
                empty_sequences += 1

            len_a = len(seq_a)
            len_b = len(seq_b)
            max_seq_a_length = max(max_seq_a_length, len_a)
            max_seq_b_length = max(max_seq_b_length, len_b)
            seq_a_truncated += int(len_a > raw_residue_limit)
            seq_b_truncated += int(len_b > raw_residue_limit)

            pair = (protein_a_id, protein_b_id)
            previous_label = seen_pairs.get(pair)
            if previous_label is not None:
                duplicate_rows += 1
                if previous_label != label:
                    conflicting_pair_labels += 1
            else:
                seen_pairs[pair] = label

            if len(examples) < 2:
                examples.append(
                    {
                        "row_index": row_index,
                        "protein_a_id": protein_a_id,
                        "protein_b_id": protein_b_id,
                        "label": label,
                        "seq_a_length": len_a,
                        "seq_b_length": len_b,
                    }
                )

    positive_count = label_counts.get("1", 0)
    negative_count = label_counts.get("0", 0)
    invalid_label_count = row_count - positive_count - negative_count
    issues: List[str] = []
    warnings: List[str] = []
    if row_count == 0:
        issues.append("CSV has no data rows")
    if invalid_label_count:
        issues.append(f"{invalid_label_count} rows have labels other than 0 or 1")
    if positive_count == 0 or negative_count == 0:
        issues.append(
            "Binary evaluation is impossible because one label class is absent "
            f"(positive={positive_count}, negative={negative_count})"
        )
    if empty_ids:
        issues.append(f"{empty_ids} rows have an empty protein identifier")
    if empty_sequences:
        issues.append(f"{empty_sequences} rows have an empty sequence")
    if malformed_rows:
        issues.append(f"{malformed_rows} rows have extra malformed CSV fields")
    if conflicting_pair_labels:
        issues.append(
            f"{conflicting_pair_labels} repeated ordered pairs have conflicting labels"
        )
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} ordered protein pairs are repeated")
    if positive_count and negative_count:
        ratio = negative_count / positive_count
        if not math.isclose(ratio, 10.0, rel_tol=0.0, abs_tol=1e-9):
            warnings.append(
                f"negative:positive ratio is {ratio:.6g}:1 rather than the expected 10:1"
            )
    else:
        ratio = None

    return {
        "file": fingerprint_file(path, hash_mode),
        "columns": list(fieldnames),
        "row_count": row_count,
        "label_counts": dict(sorted(label_counts.items())),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_prevalence": positive_count / row_count if row_count else None,
        "negative_to_positive_ratio": ratio,
        "duplicate_ordered_pair_rows": duplicate_rows,
        "conflicting_ordered_pair_labels": conflicting_pair_labels,
        "empty_identifier_rows": empty_ids,
        "empty_sequence_rows": empty_sequences,
        "malformed_rows": malformed_rows,
        "max_seq_a_length": max_seq_a_length,
        "max_seq_b_length": max_seq_b_length,
        "seq_a_rows_truncated_at_model_input": seq_a_truncated,
        "seq_b_rows_truncated_at_model_input": seq_b_truncated,
        "tokenizer_max_length": tokenizer_max_length,
        "sample_identity_rows": examples,
        "valid_for_binary_evaluation": not issues,
        "issues": issues,
        "warnings": warnings,
        "audited_at_utc": utc_now(),
    }

def atomic_write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise

def basic_environment() -> Dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "working_directory": str(Path.cwd().resolve()),
    }

def finite_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} value in predictions: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} value in predictions: {value!r}")
    return parsed

def read_prediction_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise ValueError(
                f"Unexpected prediction columns in {path}; expected "
                f"{', '.join(PREDICTION_COLUMNS)}"
            )
        yield from reader
