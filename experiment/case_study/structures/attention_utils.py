from __future__ import annotations
import importlib
import gc
import sys
from pathlib import Path
from contextlib import nullcontext
from typing import Mapping
import numpy as np

def import_runtime(args):

    model_dir = str(args.model_dir.resolve())
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    importlib.invalidate_caches()

    import torch
    from transformers import AutoTokenizer

    source_config = importlib.import_module("config")
    source_config.config.model_name = str(args.esm_dir.resolve())
    source_config.config.max_seq_len = args.max_length
    source_config.config.gradient_checkpointing = False
    source_model = importlib.import_module("model")
    return torch, AutoTokenizer, source_config.config, source_model.StructPEFTCL

def autocast_context(torch, amp: str):
    if amp == "none":
        return nullcontext()
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)

def tokenize_chain(tokenizer, sequence: str, max_length: int, padding_mode: str, torch):
    kwargs = {
        "add_special_tokens": True,
        "return_attention_mask": True,
        "return_special_tokens_mask": True,
        "return_tensors": "pt",
        "truncation": False,
    }
    if padding_mode == "max_length":
        kwargs.update({"padding": "max_length", "max_length": max_length})
    else:
        kwargs.update({"padding": False})
    encoded = tokenizer(sequence, **kwargs)
    token_positions = torch.nonzero(
        (encoded["special_tokens_mask"][0] == 0) & (encoded["attention_mask"][0] == 1),
        as_tuple=False,
    ).flatten()
    if int(token_positions.numel()) != len(sequence):
        raise RuntimeError(
            f"Tokenizer produced {int(token_positions.numel())} residue tokens for a "
            f"{len(sequence)}-residue sequence; residue mapping is unsafe."
        )
    model_inputs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    return model_inputs, token_positions

def to_device(inputs: Mapping[str, object], device):
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}

def load_checkpoint_strict(model, checkpoint_path: Path, torch):
    from checkpoints import read_checkpoint_state
    state = read_checkpoint_state(checkpoint_path)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed. Missing={result.missing_keys}; "
            f"unexpected={result.unexpected_keys}"
        )
    del state
    gc.collect()

def incoming_scores(matrix, query_positions, key_positions):

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 3:
        matrix = matrix.mean(axis=0)
    sub = matrix[np.ix_(np.asarray(query_positions, dtype=int), np.asarray(key_positions, dtype=int))]
    if not sub.size or not np.isfinite(sub).all() or (sub < 0).any():
        raise ValueError('Invalid attention matrix or residue positions')
    totals = sub.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError('Attention has no mass on valid residue keys')
    return (sub / totals).mean(axis=0)

def patch_rows(pdb_id, role, residues, scores, width=25):
    rows = []
    for start in range(0, len(residues), width):
        end = min(start + width, len(residues))
        rows.append({'pdb_id': pdb_id, 'chain_role': role, 'chain_id': residues[0].chain_id,
                     'patch_id': f'{role}{start // width + 1:04d}', 'start_1': start + 1,
                     'end_1': end, 'length': end-start, 'attention_score': float(np.mean(scores[start:end]))})
    return rows
