from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "config.py").is_file())
_sys.path.insert(0, str(_ROOT / "core"))
_sys.path.append(str(_ROOT / "experiment" / "ablation"))
_os.environ["TOPPPI_TASK"] = 'human_virus'

import argparse
import contextlib
import csv
import gc
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    PREDICTION_COLUMNS,
    REQUIRED_COLUMNS,
    atomic_write_json,
    audit_csv,
    basic_environment,
    canonical_folds,
    canonical_splits,
    checkpoint_path,
    csv_path,
    fingerprint_file,
    finite_float,
    full_file_sha256,
    timestamp_token,
    utc_now,
)

INPUT_ORDER = "CSV seq_a -> model first positional branch; CSV seq_b -> model second positional branch"
DEFAULT_SEED = 2026
VF1_C2_RECORDED = {
    "AUROC": 0.8602,
    "AUPRC": 0.5939,
    "Accuracy": 0.9359,
    "Precision": 0.8075,
    "Recall": 0.3870,
    "F1": 0.5232,
    "MCC": 0.5320,
}
VF1_C2_PRIMARY_TOLERANCE = 0.002

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_repo = _ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Audit and evaluate saved Human-virus checkpoints without retraining or "
            "modifying source data/checkpoints."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Merged binary CSV root. Default: "
            "<repo>/data/my_csv_data (the server layout shown by the user)."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--esm-dir",
        type=Path,
        default=None,
        help="Local ESM-2 model directory; auto-detected from the repository or its parent.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_ROOT / "outputs/human_virus/evaluation"
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["VF1"],
        help="Fold names separated by spaces or commas (default: VF1, the supplied selected model).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="C1Virus/C2Virus (aliases C1/C2 also accepted).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit CSV identity/classes only; imports no ML dependencies.",
    )
    mode.add_argument(
        "--check-load-only",
        action="store_true",
        help="Audit data and strict-load all selected checkpoints; run no inference.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows per fold/split (smoke tests only).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device such as cuda, cuda:0, or cpu (default: cuda).",
    )
    parser.add_argument(
        "--amp",
        choices=("off", "float16", "bfloat16"),
        default="off",
        help="Default off reproduces the historical evaluation path most closely.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--hash-mode",
        choices=("none", "quick", "full"),
        default="quick",
        help="Source-data fingerprint mode; use full for final archival runs.",
    )
    parser.add_argument(
        "--checkpoint-hash-mode",
        choices=("none", "quick", "full"),
        default="quick",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Validate and append to an interrupted prediction file.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the selected evaluator outputs; never touches source files.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many newly inferred rows.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must lie in [0, 1]")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")
    return args

def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    repo_root = args.repo_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root is not None
        else (repo_root / "data" / "human_virus").resolve()
    )
    checkpoint_dir = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir is not None
        else (repo_root / "checkpoints/human_virus").resolve()
    )
    model_dir = (
        args.model_dir.expanduser().resolve()
        if args.model_dir is not None
        else (repo_root / "core").resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()

    if args.esm_dir is not None:
        esm_dir = args.esm_dir.expanduser().resolve()
    else:
        candidates = (
            repo_root / "pretrained" / "esm2_t33_650M_UR50D",
            repo_root / "esm2_t33_650M_UR50D",
            repo_root.parent / "esm2_t33_650M_UR50D",
        )
        esm_dir = next((p.resolve() for p in candidates if p.is_dir()), candidates[-1].resolve())

    if output_dir == checkpoint_dir:
        raise ValueError(f"Refusing output directory {output_dir}: it is the checkpoint directory")
    for protected in (data_root, model_dir, esm_dir):
        if is_within(output_dir, protected):
            raise ValueError(
                f"Refusing output directory {output_dir}: it is inside protected input {protected}"
            )

    return {
        "repo_root": repo_root,
        "data_root": data_root,
        "checkpoint_dir": checkpoint_dir,
        "model_dir": model_dir,
        "esm_dir": esm_dir,
        "output_dir": output_dir,
    }

def unique_json_path(directory: Path, prefix: str) -> Path:
    return directory / f"{prefix}_{timestamp_token()}_{uuid.uuid4().hex[:8]}.json"

def source_code_fingerprints(model_dir: Path) -> Dict[str, Any]:
    paths = {
        "evaluator": Path(__file__).resolve(),
        "common": SCRIPT_DIR / "common.py",
        "model": model_dir / "model.py",
        "config": model_dir / "config.py",
        "layers": model_dir / "layers.py",
    }
    result: Dict[str, Any] = {}
    for name, path in paths.items():
        if path.is_file():
            result[name] = {
                "path": str(path.resolve()),
                "sha256": full_file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        else:
            result[name] = {"path": str(path.resolve()), "missing": True}
    return result

def git_revision(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None

def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def dependency_versions() -> Dict[str, Optional[str]]:
    return {
        name: package_version(name)
        for name in (
            "torch",
            "torch-geometric",
            "transformers",
            "peft",
            "numpy",
            "scikit-learn",
        )
    }

def load_ml_modules(model_dir: Path, esm_dir: Path) -> Dict[str, Any]:
    missing_files = [
        str(path)
        for path in (model_dir / "model.py", model_dir / "config.py", model_dir / "layers.py")
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError("Missing model source files: " + ", ".join(missing_files))
    if not esm_dir.is_dir():
        raise FileNotFoundError(
            f"Local ESM-2 directory not found: {esm_dir}. Pass --esm-dir explicitly."
        )

    for module_name in ("config", "layers", "model"):
        existing = sys.modules.get(module_name)
        if existing is not None:
            module_file = Path(getattr(existing, "__file__", "")).resolve()
            expected = (model_dir / f"{module_name}.py").resolve()
            if module_file != expected:
                raise RuntimeError(
                    f"Python module collision: {module_name!r} already came from {module_file}, "
                    f"expected {expected}. Run the evaluator in a fresh Python process."
                )

    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing ML dependencies. Activate the original DeepGNHV/TopPPI "
            "environment with torch, torch-geometric, transformers, peft, numpy, and "
            "scikit-learn installed."
        ) from exc

    config_module = importlib.import_module("config")
    if Path(config_module.__file__).resolve() != (model_dir / "config.py").resolve():
        raise RuntimeError(f"Loaded the wrong config.py: {config_module.__file__}")
    config = config_module.config
    config.model_name = str(esm_dir)

    model_module = importlib.import_module("model")
    if Path(model_module.__file__).resolve() != (model_dir / "model.py").resolve():
        raise RuntimeError(f"Loaded the wrong model.py: {model_module.__file__}")

    tokenizer = AutoTokenizer.from_pretrained(str(esm_dir), local_files_only=True)
    model = model_module.TopPPI()
    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "config": config,
    }

def load_state_strict(model: Any, path: Path, torch: Any) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        kwargs["weights_only"] = True
    if "mmap" in parameters:
        kwargs["mmap"] = True
    started = time.perf_counter()
    payload = torch.load(str(path), **kwargs)
    if isinstance(payload, Mapping) and payload.get('format') == 'topppi-shared-backbone-v1':
        from checkpoints import read_checkpoint_state
        payload = {'state_dict': read_checkpoint_state(path)}
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
        container = "state_dict"
    else:
        state = payload
        container = "raw_mapping"
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {path} did not contain a state-dict mapping")

    incompatible = model.load_state_dict(state, strict=True)
    missing = list(getattr(incompatible, "missing_keys", ()))
    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            f"Strict load unexpectedly reported missing={missing}, unexpected={unexpected}"
        )
    elapsed = time.perf_counter() - started
    state_key_count = len(state)
    del state
    del payload
    gc.collect()
    return {
        "strict": True,
        "container": container,
        "state_key_count": state_key_count,
        "missing_keys": [],
        "unexpected_keys": [],
        "load_seconds": elapsed,
    }

def resolve_device(torch: Any, requested: str) -> Any:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Run on the GPU host, or pass "
            "--device cpu only for a very small diagnostic (full CPU evaluation is impractical)."
        )
    return device

def torch_environment(torch: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(properties.major_minor)
                    if hasattr(properties, "major_minor")
                    else [properties.major, properties.minor],
                }
            )
    return result

def set_reproducibility(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prediction_paths(
    output_dir: Path, fold: str, split: str, limit: Optional[int]
) -> Tuple[Path, Path, Path]:
    tag = "full" if limit is None else f"limit{limit}"
    stem = f"{fold}_{split}_{tag}"
    return (
        output_dir / "predictions" / f"{stem}.csv",
        output_dir / "states" / f"{stem}.json",
        output_dir / "metrics" / f"{stem}.json",
    )

def state_identity(
    *,
    fold: str,
    split: str,
    source_audit: Mapping[str, Any],
    checkpoint_fingerprint: Mapping[str, Any],
    limit: Optional[int],
    target_count: int,
    threshold: float,
    tokenizer_max_length: int,
) -> Dict[str, Any]:
    source_file = source_audit["file"]
    return {
        "fold": fold,
        "split": split,
        "source_path": source_file["path"],
        "source_size_bytes": source_file["size_bytes"],
        "source_mtime_utc": source_file["mtime_utc"],
        "source_fingerprint": dict(source_file),
        "checkpoint_path": checkpoint_fingerprint["path"],
        "checkpoint_size_bytes": checkpoint_fingerprint["size_bytes"],
        "checkpoint_mtime_utc": checkpoint_fingerprint["mtime_utc"],
        "checkpoint_fingerprint": dict(checkpoint_fingerprint),
        "limit": limit,
        "target_count": target_count,
        "threshold": threshold,
        "tokenizer_max_length": tokenizer_max_length,
        "input_order": INPUT_ORDER,
    }

def compare_state_identity(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    differences = []
    for key, value in expected.items():
        if existing.get(key) != value:
            differences.append(f"{key}: existing={existing.get(key)!r}, expected={value!r}")
    if differences:
        raise ValueError(
            "Cannot resume because run identity changed:\n  " + "\n  ".join(differences)
        )

def validate_existing_predictions(
    prediction_file: Path,
    source_file: Path,
    *,
    target_count: int,
    threshold: float,
) -> int:

    count = 0
    with prediction_file.open("r", encoding="utf-8", newline="") as pred_handle, source_file.open(
        "r", encoding="utf-8-sig", newline=""
    ) as source_handle:
        pred_reader = csv.DictReader(pred_handle)
        source_reader = csv.DictReader(source_handle)
        if tuple(pred_reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise ValueError(
                f"Prediction header mismatch in {prediction_file}; refusing unsafe resume"
            )
        for expected_index, pred_row in enumerate(pred_reader):
            if expected_index >= target_count:
                raise ValueError(
                    f"Existing predictions exceed this run's target_count={target_count}"
                )
            try:
                source_row = next(source_reader)
            except StopIteration as exc:
                raise ValueError("Predictions contain more rows than the source CSV") from exc
            try:
                stored_index = int(pred_row["row_index"])
                stored_label = int(pred_row["label"])
                source_label = int(source_row["label"])
                stored_prediction = int(pred_row["prediction"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed prediction row {expected_index} in {prediction_file}"
                ) from exc
            probability = finite_float(pred_row["probability"], "probability")
            finite_float(pred_row["logit"], "logit")
            expected_prediction = int(probability >= threshold)
            if (
                stored_index != expected_index
                or pred_row["protein_a_id"] != source_row["protein_a_id"]
                or pred_row["protein_b_id"] != source_row["protein_b_id"]
                or stored_label != source_label
                or stored_prediction != expected_prediction
            ):
                raise ValueError(
                    f"Prediction/source identity mismatch at row {expected_index}; "
                    "refusing unsafe resume"
                )
            count += 1
    return count

def source_batches(
    source_file: Path,
    *,
    start_index: int,
    target_count: int,
    batch_size: int,
) -> Iterator[List[Tuple[int, Dict[str, str]]]]:
    batch: List[Tuple[int, Dict[str, str]]] = []
    with source_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Source file lost required columns: {missing}")
        for row_index, row in enumerate(reader):
            if row_index < start_index:
                continue
            if row_index >= target_count:
                break
            batch.append((row_index, row))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def autocast_context(torch: Any, device: Any, amp: str) -> Any:
    if amp == "off":
        return contextlib.nullcontext()
    if device.type != "cuda":
        raise ValueError("--amp float16/bfloat16 is supported here only with CUDA")
    dtype = torch.float16 if amp == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)

def infer_one_source(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    config: Any,
    source_file: Path,
    prediction_file: Path,
    state_file: Path,
    state_base: Mapping[str, Any],
    target_count: int,
    batch_size: int,
    threshold: float,
    amp: str,
    resume: bool,
    overwrite: bool,
    progress_every: int,
) -> Dict[str, Any]:
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for path in (prediction_file, state_file):
            if path.exists():
                path.unlink()

    if prediction_file.exists() and not resume:
        raise FileExistsError(
            f"Prediction output already exists: {prediction_file}. Use --resume after an "
            "interruption or --overwrite to replace this exact result."
        )

    if prediction_file.exists():
        if not state_file.is_file():
            raise FileNotFoundError(
                f"Resume state is missing for {prediction_file}; refusing to guess identity"
            )
        with state_file.open("r", encoding="utf-8") as handle:
            existing_state = json.load(handle)
        compare_state_identity(existing_state, state_base)
        completed_before = validate_existing_predictions(
            prediction_file, source_file, target_count=target_count, threshold=threshold
        )
        state_rows = existing_state.get("rows_written")
        if not isinstance(state_rows, int) or state_rows < 0:
            raise ValueError(f"Invalid rows_written value in {state_file}: {state_rows!r}")
        if state_rows > completed_before:
            raise ValueError(
                f"Resume state says {state_rows} rows, but the validated prediction file "
                f"contains only {completed_before}"
            )

        file_mode = "a"
    else:
        completed_before = 0
        file_mode = "w"

    state = dict(state_base)
    state.update(
        {
            "status": "running",
            "rows_written": completed_before,
            "prediction_file": str(prediction_file.resolve()),
            "started_or_resumed_at_utc": utc_now(),
        }
    )
    atomic_write_json(state_file, state)

    newly_written = 0
    started = time.perf_counter()
    last_report = 0
    model.eval()
    with prediction_file.open(file_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        if file_mode == "w":
            writer.writeheader()
            handle.flush()

        for batch in source_batches(
            source_file,
            start_index=completed_before,
            target_count=target_count,
            batch_size=batch_size,
        ):
            seq_a = ["".join(row["seq_a"].split()) for _, row in batch]
            seq_b = ["".join(row["seq_b"].split()) for _, row in batch]

            branch1 = tokenizer(
                seq_a,
                truncation=True,
                max_length=int(config.max_seq_len),
                padding="max_length",
                return_tensors="pt",
            )
            branch2 = tokenizer(
                seq_b,
                truncation=True,
                max_length=int(config.max_seq_len),
                padding="max_length",
                return_tensors="pt",
            )
            branch1_inputs = {
                "input_ids": branch1["input_ids"].to(device, non_blocking=True),
                "attention_mask": branch1["attention_mask"].to(device, non_blocking=True),
            }
            branch2_inputs = {
                "input_ids": branch2["input_ids"].to(device, non_blocking=True),
                "attention_mask": branch2["attention_mask"].to(device, non_blocking=True),
            }

            with torch.inference_mode():
                with autocast_context(torch, device, amp):
                    logits, _, _ = model(branch1_inputs, branch2_inputs)
                logits_cpu = logits.detach().float().view(-1).cpu()
                probabilities_cpu = torch.sigmoid(logits_cpu)

            for (row_index, row), logit, probability in zip(
                batch, logits_cpu.tolist(), probabilities_cpu.tolist()
            ):
                prediction = int(probability >= threshold)
                writer.writerow(
                    {
                        "row_index": row_index,
                        "protein_a_id": row["protein_a_id"],
                        "protein_b_id": row["protein_b_id"],
                        "label": int(row["label"]),
                        "logit": format(float(logit), ".10g"),
                        "probability": format(float(probability), ".10g"),
                        "prediction": prediction,
                    }
                )
                newly_written += 1

            handle.flush()
            state["rows_written"] = completed_before + newly_written
            state["updated_at_utc"] = utc_now()
            atomic_write_json(state_file, state)

            if newly_written - last_report >= progress_every:
                elapsed = max(time.perf_counter() - started, 1e-9)
                rate = newly_written / elapsed
                print(
                    f"  {state['fold']} {state['split']}: "
                    f"{completed_before + newly_written}/{target_count} rows "
                    f"({rate:.3f} new rows/s)",
                    flush=True,
                )
                last_report = newly_written

            del branch1_inputs, branch2_inputs, branch1, branch2, logits_cpu, probabilities_cpu

    elapsed = time.perf_counter() - started
    total_rows = validate_existing_predictions(
        prediction_file, source_file, target_count=target_count, threshold=threshold
    )
    if total_rows != target_count:
        raise RuntimeError(
            f"Inference stopped at {total_rows} rows but target_count={target_count}"
        )
    state.update(
        {
            "status": "predictions_complete",
            "rows_written": total_rows,
            "completed_at_utc": utc_now(),
            "new_rows_this_invocation": newly_written,
            "inference_seconds_this_invocation": elapsed,
            "new_rows_per_second": newly_written / elapsed if elapsed > 0 else None,
        }
    )
    atomic_write_json(state_file, state)
    return state

def compute_metrics(prediction_file: Path, threshold: float) -> Tuple[Dict[str, Any], List[str]]:
    try:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Metrics require numpy and scikit-learn in the evaluation environment"
        ) from exc

    labels: List[int] = []
    probabilities: List[float] = []
    with prediction_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise ValueError(f"Unexpected prediction header in {prediction_file}")
        for row in reader:
            labels.append(int(row["label"]))
            probabilities.append(finite_float(row["probability"], "probability"))

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    warnings: List[str] = []
    unique_labels = np.unique(y_true)
    if unique_labels.size == 2:
        auroc: Optional[float] = float(roc_auc_score(y_true, y_prob))
        auprc: Optional[float] = float(average_precision_score(y_true, y_prob))
    else:
        auroc = None
        auprc = None
        warnings.append(
            "AUROC/AUPRC were not reported because this evaluated subset contains only "
            f"label(s) {unique_labels.tolist()}; this is expected for some small --limit smoke tests."
        )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    result: Dict[str, Any] = {
        "n": int(y_true.size),
        "positive_count": int((y_true == 1).sum()),
        "negative_count": int((y_true == 0).sum()),
        "positive_prevalence": float((y_true == 1).mean()) if y_true.size else None,
        "threshold": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "AUROC": auroc,
        "AUPRC": auprc,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": specificity,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    return result, warnings

def print_metrics(fold: str, split: str, metrics: Mapping[str, Any], partial: bool) -> None:
    suffix = " [PARTIAL/SMOKE TEST]" if partial else ""
    print(f"\n{fold} {split}{suffix}")
    for name in (
        "AUROC",
        "AUPRC",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "MCC",
    ):
        value = metrics.get(name)
        display = "NA" if value is None else f"{value:.6f}"
        print(f"  {name:12s} {display}")

def vf1_c2_reproduction_gate(
    fold: str, split: str, metrics: Mapping[str, Any], partial: bool
) -> Optional[Dict[str, Any]]:
    if fold != "VF1" or split != "C2Virus" or partial:
        return None
    differences: Dict[str, Optional[float]] = {}
    for metric, recorded in VF1_C2_RECORDED.items():
        observed = metrics.get(metric)
        differences[metric] = None if observed is None else float(observed) - recorded
    primary_pass = all(
        differences[metric] is not None
        and abs(float(differences[metric])) <= VF1_C2_PRIMARY_TOLERANCE
        for metric in ("AUROC", "AUPRC")
    )
    return {
        "status": "pass" if primary_pass else "fail",
        "primary_metrics": ["AUROC", "AUPRC"],
        "absolute_tolerance": VF1_C2_PRIMARY_TOLERANCE,
        "recorded_metrics": dict(VF1_C2_RECORDED),
        "observed_minus_recorded": differences,
        "instruction": (
            "Proceed to the remaining folds only if this gate passes. If it fails, "
            "check data/code/checkpoint identity, input order, software versions, and AMP."
        ),
    }

def run(args: argparse.Namespace) -> Path:
    paths = resolve_paths(args)
    folds = canonical_folds(args.folds)
    splits = canonical_splits(args.splits)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = unique_json_path(output_dir / "manifests", "run")
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "evaluate_human_virus.py",
        "status": "started",
        "started_at_utc": utc_now(),
        "mode": (
            "audit_only"
            if args.audit_only
            else "check_load_only"
            if args.check_load_only
            else "evaluate"
        ),
        "input_order_contract": INPUT_ORDER,
        "arguments": {
            "folds": folds,
            "splits": splits,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "threshold": args.threshold,
            "device": args.device,
            "amp": args.amp,
            "seed": args.seed,
            "hash_mode": args.hash_mode,
            "checkpoint_hash_mode": args.checkpoint_hash_mode,
            "resume": args.resume,
            "overwrite": args.overwrite,
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "environment": basic_environment(),
        "package_versions": dependency_versions(),
        "git_revision": git_revision(paths["repo_root"]),
        "source_code": source_code_fingerprints(paths["model_dir"]),
        "data_audits": {},
        "checkpoint_fingerprints": {},
        "checkpoint_load_checks": {},
        "results": [],
    }
    atomic_write_json(manifest_path, manifest)

    try:
        print(f"Input-order contract: {INPUT_ORDER}")
        print(f"Data root: {paths['data_root']}")
        print(f"Checkpoint directory: {paths['checkpoint_dir']}", flush=True)
        print(f"Model source directory: {paths['model_dir']}", flush=True)

        for fold in folds:
            for split in splits:
                source_file = csv_path(paths["data_root"], fold, split)
                print(f"Auditing {fold} {split}: {source_file}", flush=True)
                audit = audit_csv(
                    source_file,
                    hash_mode=args.hash_mode,
                    tokenizer_max_length=800,
                )
                manifest["data_audits"][f"{fold}:{split}"] = audit
                if not audit["valid_for_binary_evaluation"]:
                    raise ValueError(
                        f"Dataset audit failed for {fold} {split}: "
                        + "; ".join(audit["issues"])
                    )
                print(
                    f"  rows={audit['row_count']:,}, positive={audit['positive_count']:,}, "
                    f"negative={audit['negative_count']:,}",
                    flush=True,
                )

        for fold in folds:
            ckpt = checkpoint_path(paths["checkpoint_dir"], fold)
            if not ckpt.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            manifest["checkpoint_fingerprints"][fold] = fingerprint_file(
                ckpt, args.checkpoint_hash_mode
            )

        audit_path = unique_json_path(output_dir / "audits", "data_audit")
        atomic_write_json(
            audit_path,
            {
                "schema_version": 1,
                "created_at_utc": utc_now(),
                "data_root": str(paths["data_root"]),
                "input_order_contract": INPUT_ORDER,
                "folds": folds,
                "splits": splits,
                "audits": manifest["data_audits"],
            },
        )
        manifest["audit_report"] = str(audit_path)

        if args.audit_only:
            manifest["status"] = "completed"
            manifest["completed_at_utc"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            print(f"Audit completed: {audit_path}")
            return manifest_path

        ml = load_ml_modules(paths["model_dir"], paths["esm_dir"])
        torch = ml["torch"]
        tokenizer = ml["tokenizer"]
        model = ml["model"]
        config = ml["config"]
        manifest["torch_environment"] = torch_environment(torch)
        manifest["model_config"] = {
            "model_name": str(config.model_name),
            "max_seq_len": int(config.max_seq_len),
            "embedding_dim": int(config.embedding_dim),
            "gat_hidden_dim": int(config.gat_hidden_dim),
            "use_lora": bool(config.use_lora),
            "lora_r": int(config.lora_r),
            "lora_alpha": int(config.lora_alpha),
            "lora_target_modules": list(config.lora_target_modules),
            "contact_head_layer": int(config.contact_head_layer),
            "contact_head_indices": list(config.contact_head_indices),
            "contact_threshold": float(config.contact_threshold),
            "knn_k": int(config.knn_k),
        }
        if int(config.max_seq_len) != 800:
            raise ValueError(
                f"Expected config.max_seq_len=800 from the saved experiment, found {config.max_seq_len}"
            )

        set_reproducibility(args.seed, torch)

        if args.check_load_only:
            for fold in folds:
                ckpt = checkpoint_path(paths["checkpoint_dir"], fold)
                print(f"Strict-loading {fold}: {ckpt}", flush=True)
                manifest["checkpoint_load_checks"][fold] = load_state_strict(
                    model, ckpt, torch
                )
                atomic_write_json(manifest_path, manifest)
            manifest["status"] = "completed"
            manifest["completed_at_utc"] = utc_now()
            atomic_write_json(manifest_path, manifest)
            print(f"All selected checkpoints strict-loaded successfully: {manifest_path}")
            return manifest_path

        device = resolve_device(torch, args.device)
        if args.amp != "off" and device.type != "cuda":
            raise ValueError("Mixed precision is only supported with a CUDA device")

        model_is_on_device = False
        for fold in folds:
            ckpt = checkpoint_path(paths["checkpoint_dir"], fold)
            print(f"Strict-loading {fold}: {ckpt}", flush=True)
            load_check = load_state_strict(model, ckpt, torch)
            manifest["checkpoint_load_checks"][fold] = load_check
            if not model_is_on_device:
                model.to(device)
                model_is_on_device = True
            model.eval()
            atomic_write_json(manifest_path, manifest)

            for split in splits:
                source_file = csv_path(paths["data_root"], fold, split)
                source_audit = manifest["data_audits"][f"{fold}:{split}"]
                source_count = int(source_audit["row_count"])
                target_count = (
                    source_count if args.limit is None else min(source_count, args.limit)
                )
                prediction_file, state_file, metrics_file = prediction_paths(
                    output_dir, fold, split, args.limit
                )
                if args.overwrite and metrics_file.exists():
                    metrics_file.unlink()
                elif metrics_file.exists() and not args.resume:
                    raise FileExistsError(
                        f"Metric output already exists: {metrics_file}. Use --resume to "
                        "validate/reuse its predictions or --overwrite to restart this exact result."
                    )
                if args.resume and metrics_file.exists() and not prediction_file.exists():
                    raise FileNotFoundError(
                        f"Metric file exists but its prediction file is missing: {prediction_file}; "
                        "refusing an ambiguous resume"
                    )

                identity = state_identity(
                    fold=fold,
                    split=split,
                    source_audit=source_audit,
                    checkpoint_fingerprint=manifest["checkpoint_fingerprints"][fold],
                    limit=args.limit,
                    target_count=target_count,
                    threshold=args.threshold,
                    tokenizer_max_length=int(config.max_seq_len),
                )
                print(
                    f"Evaluating {fold} {split}: target={target_count:,}, "
                    f"output={prediction_file}",
                    flush=True,
                )
                state = infer_one_source(
                    model=model,
                    tokenizer=tokenizer,
                    torch=torch,
                    device=device,
                    config=config,
                    source_file=source_file,
                    prediction_file=prediction_file,
                    state_file=state_file,
                    state_base=identity,
                    target_count=target_count,
                    batch_size=args.batch_size,
                    threshold=args.threshold,
                    amp=args.amp,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    progress_every=args.progress_every,
                )
                metrics, metric_warnings = compute_metrics(
                    prediction_file, args.threshold
                )
                is_partial = target_count < source_count
                metrics_payload = {
                    "schema_version": 1,
                    "created_at_utc": utc_now(),
                    "fold": fold,
                    "split": split,
                    "run_tag": "full" if args.limit is None else f"limit{args.limit}",
                    "is_partial": is_partial,
                    "source_row_count": source_count,
                    "evaluated_row_count": target_count,
                    "input_order_contract": INPUT_ORDER,
                    "source_file": source_audit["file"],
                    "checkpoint_file": manifest["checkpoint_fingerprints"][fold],
                    "prediction_file": str(prediction_file.resolve()),
                    "state_file": str(state_file.resolve()),
                    "amp": args.amp,
                    "device": str(device),
                    "batch_size": args.batch_size,
                    "metrics": metrics,
                    "warnings": metric_warnings,
                    "vf1_c2_reproduction_gate": vf1_c2_reproduction_gate(
                        fold, split, metrics, is_partial
                    ),
                    "selection_caveat": (
                        "C2Virus was used for checkpoint selection in the historical training "
                        "script; these metrics are selection-set estimates, not an independent "
                        "unseen-virus test."
                        if split == "C2Virus"
                        else None
                    ),
                }
                atomic_write_json(metrics_file, metrics_payload)
                state["status"] = "completed"
                state["metrics_file"] = str(metrics_file.resolve())
                state["completed_at_utc"] = utc_now()
                atomic_write_json(state_file, state)
                manifest["results"].append(metrics_payload)
                atomic_write_json(manifest_path, manifest)
                print_metrics(fold, split, metrics, is_partial)
                gate = metrics_payload["vf1_c2_reproduction_gate"]
                if gate is not None:
                    print(
                        f"  VF1-C2 reproduction gate: {gate['status'].upper()} "
                        f"(AUROC/AUPRC tolerance ±{gate['absolute_tolerance']})"
                    )

            if device.type == "cuda":
                torch.cuda.empty_cache()

        manifest["status"] = "completed"
        manifest["completed_at_utc"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        print(f"Evaluation completed. Run manifest: {manifest_path}")
        return manifest_path
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = utc_now()
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(manifest_path, manifest)
        print(f"FAILED. Diagnostic manifest: {manifest_path}", file=sys.stderr)
        raise

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = run(args)
    print(f"Manifest: {manifest_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
