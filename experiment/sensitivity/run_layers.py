import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "config.py").is_file())
_sys.path.insert(0, str(_ROOT / "core"))
_sys.path.append(str(_ROOT / "experiment" / "ablation"))
_os.environ["TOPPPI_TASK"] = 'yeast'

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from config import config
from loss import DualBranchLoss
from model_ablation import ABLATION_VARIANTS, available_variants, build_ablation_model, resolve_model_name

class YeastDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=800):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_a = row.get("seq_a") or row.get("protein_a") or row.get("sequence_a")
        seq_b = row.get("seq_b") or row.get("protein_b") or row.get("sequence_b")
        label = float(row["label"])

        enc_a = self.tokenizer(str(seq_a), padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        enc_b = self.tokenizer(str(seq_b), padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def is_distributed_run(): return int(os.environ.get("WORLD_SIZE", "1")) > 1

def setup_device(use_ddp):
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), local_rank
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0

def cleanup_ddp(use_ddp):
    if use_ddp and dist.is_initialized(): dist.destroy_process_group()

def is_main_process(use_ddp, local_rank): return (not use_ddp) or local_rank == 0

def reduce_mean_tensor(value, device, use_ddp):
    tensor = torch.tensor(float(value), device=device)
    if use_ddp:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()

def compute_metrics(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs > 0.5).astype(int)
    try:
        auc, aupr = roc_auc_score(labels, probs), average_precision_score(labels, probs)
    except ValueError:
        auc, aupr = 0.5, 0.0
    acc, f1 = accuracy_score(labels, preds), f1_score(labels, preds, zero_division=0)
    prec, rec = precision_score(labels, preds, zero_division=0), recall_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"AUC": float(auc), "AUPR": float(aupr), "Accuracy": float(acc), "F1": float(f1),
            "Precision": float(prec), "Recall": float(rec), "MCC": float(mcc), "Specificity": float(specificity)}

def evaluate(model, dataloader, device, use_ddp=False):
    if use_ddp and dist.get_rank() != 0: return None
    eval_model = model.module if use_ddp and hasattr(model, "module") else model
    eval_model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs_a = {"input_ids": batch["input_ids_a"].to(device), "attention_mask": batch["attention_mask_a"].to(device)}
            inputs_b = {"input_ids": batch["input_ids_b"].to(device), "attention_mask": batch["attention_mask_b"].to(device)}
            labels = batch["label"].to(device)
            with autocast(enabled=config.use_fp16 and device.type == "cuda"):
                logits, _, _ = eval_model(inputs_a, inputs_b)
            probs = torch.sigmoid(logits).view(-1).detach().cpu().float().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
    return compute_metrics(all_labels, all_probs)

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, use_ddp, local_rank, epoch):
    model.train()
    total_loss, total_steps = 0.0, 0
    if isinstance(dataloader.sampler, DistributedSampler): dataloader.sampler.set_epoch(epoch)
    iterator = tqdm(dataloader, desc=f"Ep {epoch + 1}") if is_main_process(use_ddp, local_rank) else dataloader
    for batch in iterator:
        inputs_a = {"input_ids": batch["input_ids_a"].to(device), "attention_mask": batch["attention_mask_a"].to(device)}
        inputs_b = {"input_ids": batch["input_ids_b"].to(device), "attention_mask": batch["attention_mask_b"].to(device)}
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config.use_fp16 and device.type == "cuda"):
            logits, proj, extra_scores = model(inputs_a, inputs_b)
            loss_out = criterion(logits, proj, labels)
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
            if use_ddp:
                loss = loss + (proj * 0).sum() + (extra_scores[0] * 0).sum() + (extra_scores[1] * 0).sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_steps += 1
        if is_main_process(use_ddp, local_rank): iterator.set_postfix(loss=loss.item())
    return reduce_mean_tensor(total_loss / max(total_steps, 1), device, use_ddp)

def format_metrics(metrics):
    return f"AUC: {metrics['AUC']*100:.2f} | AUPR: {metrics['AUPR']*100:.2f} | Acc: {metrics['Accuracy']*100:.2f} | F1: {metrics['F1']*100:.2f} | MCC: {metrics['MCC']*100:.2f}"

def save_summary_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

def resolve_path(path_value, model_dir):
    path = Path(path_value)
    if path.is_absolute(): return path
    candidates = [model_dir / path, model_dir.parent / path, model_dir.parent / "data" / path, Path.cwd() / path]
    return next((p for p in candidates if p.exists()), path)

def get_fixed_fold_files(fold_dir, k_folds):
    pairs = []
    for fold in range(1, k_folds + 1):
        train_file, valid_file = fold_dir / f"train{fold}.csv", fold_dir / f"valid{fold}.csv"
        if not train_file.exists() or not valid_file.exists():
            raise FileNotFoundError(f"Missing fixed split files for fold {fold}")
        pairs.append((train_file, valid_file))
    return pairs

def get_state_dict(model, use_ddp): return model.module.state_dict() if use_ddp else model.state_dict()

def checkpoint_paths(output_dir, dataset_name, variant, fold):
    stem = f"{dataset_name}_{variant}_fold_{fold}"
    return {"best": Path(output_dir) / f"{stem}_best.pth", "last": Path(output_dir) / f"{stem}_last.pt", "done": Path(output_dir) / f"{stem}_done.json"}

def save_last_checkpoint(path, model, optimizer, scaler, use_ddp, variant, fold, epoch, args, best_fold_f1, best_fold_metrics, summary_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": variant, "fold": fold, "epoch": epoch, "model_state_dict": get_state_dict(model, use_ddp),
                "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_fold_f1": best_fold_f1, "best_fold_metrics": best_fold_metrics, "summary_rows": summary_rows, "args": vars(args)}, path)

def torch_load_checkpoint(path, device):
    try: return torch.load(path, map_location=device, weights_only=False)
    except TypeError: return torch.load(path, map_location=device)

def load_last_checkpoint(path, model, optimizer, scaler, device, use_ddp):
    checkpoint = torch_load_checkpoint(path, device)
    target_model = model.module if use_ddp and hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"): scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint

def save_done_marker(path, variant, fold, best_fold_f1, best_fold_metrics, completed_epochs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"variant": variant, "fold": fold, "best_fold_f1": float(best_fold_f1),
                                "best_fold_metrics": best_fold_metrics, "completed_epochs": int(completed_epochs)}, indent=2), encoding="utf-8")

def load_done_marker(path): return json.loads(path.read_text(encoding="utf-8"))

def build_fold_pairs(args, model_dir, local_rank, use_ddp):
    if args.split_source == "fixed":
        fold_dir = resolve_path(args.fold_dir, model_dir)
        fold_pairs = get_fixed_fold_files(fold_dir, args.k_folds)
        if is_main_process(use_ddp, local_rank): print(f"Using fixed fold files from: {fold_dir}")
        return fold_pairs
    data_path = resolve_path(args.data_path, model_dir)
    df = pd.read_csv(data_path)
    if args.split_source == "single":
        stratify_col = df['label'].astype(int) if 'label' in df.columns else None
        train_df, val_df = train_test_split(df, test_size=args.val_ratio, random_state=args.seed, stratify=stratify_col)
        return [(train_df, val_df)]
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    return [(df.iloc[train_idx], df.iloc[val_idx]) for train_idx, val_idx in kf.split(df)]

def get_fold_dataframes(args, train_source, val_source, local_rank, use_ddp):
    if args.split_source == "fixed": return pd.read_csv(train_source), pd.read_csv(val_source)
    return train_source, val_source

def _run_single_experiment(args, use_ddp, device, local_rank):
    set_seed(args.seed)

    config.contact_head_layer = int(args.attention_layer)
    print(f"[Override] config.contact_head_layer = {config.contact_head_layer} (zero-based index)")

    if args.knn_k is not None:
        config.knn_k = args.knn_k
        print(f"[Override] config.knn_k = {config.knn_k}")
    if args.contact_threshold is not None:
        config.contact_threshold = args.contact_threshold
        print(f"[Override] config.contact_threshold = {config.contact_threshold}")
    if args.gat_heads is not None:
        config.gat_heads = args.gat_heads
        print(f"[Override] config.gat_heads = {config.gat_heads}")

    try:
        if is_main_process(use_ddp, local_rank):
            print(f"Running Yeast Hyperparameter Sensitivity Experiment | Model: {args.variant} | Device: {device} | DDP: {use_ddp}")

        model_dir = Path(__file__).resolve().parent
        tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(config.model_name))
        fold_pairs = build_fold_pairs(args, model_dir, local_rank, use_ddp)

        fold_results, summary_rows = [], []
        output_dir = Path(args.output_dir)
        epochs_csv = output_dir / f"{args.dataset_name}_{args.variant}_epochs.csv"

        if args.resume and epochs_csv.exists():
            try:
                summary_rows = pd.read_csv(epochs_csv).to_dict("records")
            except Exception as exc:
                if is_main_process(use_ddp, local_rank):
                    print(f"[resume warning] Could not read existing epoch CSV {epochs_csv}: {exc}")

        for fold, (train_source, val_source) in enumerate(fold_pairs):
            fold_id = fold + 1
            if args.only_fold is not None and fold_id != args.only_fold: continue

            paths = checkpoint_paths(output_dir, args.dataset_name, args.variant, fold_id)
            if args.resume and paths["done"].exists():
                done_payload = load_done_marker(paths["done"])
                completed_epochs = int(done_payload.get("completed_epochs", 0))
                if completed_epochs == args.epochs:
                    if is_main_process(use_ddp, local_rank):
                        print(f"Skipping completed fold {fold_id} ({completed_epochs}/{args.epochs} epochs); "
                              f"best F1={done_payload['best_fold_f1'] * 100:.2f}%")
                        fold_results.append(done_payload["best_fold_metrics"])
                    if use_ddp: dist.barrier()
                    continue
                if completed_epochs > args.epochs:
                    raise RuntimeError(
                        f"Fold {fold_id} already completed {completed_epochs} epochs, but this run requests "
                        f"{args.epochs}. Use a new --output-dir or restore the intended epoch count."
                    )
                if is_main_process(use_ddp, local_rank):
                    print(f"[resume] Fold {fold_id} has {completed_epochs}/{args.epochs} completed epochs; "
                          "will continue from its last checkpoint.")

            if is_main_process(use_ddp, local_rank):
                print(f"\n{'=' * 20} Fold {fold_id}/{len(fold_pairs)} {'=' * 20}")

            train_df, val_df = get_fold_dataframes(args, train_source, val_source, local_rank, use_ddp)
            train_ds, val_ds = YeastDataset(train_df, tokenizer, config.max_seq_len), YeastDataset(val_df, tokenizer, config.max_seq_len)
            train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

            model = build_ablation_model(args.variant).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            if use_ddp:
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
                model._set_static_graph()

            criterion = DualBranchLoss(alpha=config.alpha_supcon, beta=config.beta_focal, temperature=config.temperature, focal_alpha=config.focal_alpha, focal_gamma=getattr(config, "focal_gamma", 2.0)).to(device)
            scaler = GradScaler(enabled=config.use_fp16 and device.type == "cuda")
            best_fold_f1, best_fold_metrics, start_epoch = -1.0, None, 0

            if args.resume and paths["last"].exists():
                checkpoint = load_last_checkpoint(paths["last"], model, optimizer, scaler, device, use_ddp)
                start_epoch, best_fold_f1, best_fold_metrics = int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_fold_f1", -1.0)), checkpoint.get("best_fold_metrics")
                if is_main_process(use_ddp, local_rank):

                    existing_keys = {
                        (str(row.get("variant", args.variant)), int(row.get("fold", -1)), int(row.get("epoch", -1)))
                        for row in summary_rows
                    }
                    for row in checkpoint.get("summary_rows", []):
                        key = (str(row.get("variant", args.variant)), int(row.get("fold", -1)), int(row.get("epoch", -1)))
                        if key not in existing_keys:
                            summary_rows.append(row)
                            existing_keys.add(key)
                    print(f"Resumed fold {fold_id} from epoch {start_epoch}")

            for epoch in range(start_epoch, args.epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_ddp, local_rank, epoch)
                metrics = evaluate(model, val_loader, device, use_ddp=use_ddp)

                if is_main_process(use_ddp, local_rank):
                    print(f"   Loss: {train_loss:.4f} | Val -> {format_metrics(metrics)}")
                    summary_rows.append({"variant": args.variant, "fold": fold_id, "epoch": epoch + 1, "train_loss": train_loss, **metrics})
                    if metrics["F1"] > best_fold_f1:
                        best_fold_f1, best_fold_metrics = metrics["F1"], metrics
                        output_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(get_state_dict(model, use_ddp), paths["best"])
                    save_last_checkpoint(paths["last"], model, optimizer, scaler, use_ddp, args.variant, fold_id, epoch, args, best_fold_f1, best_fold_metrics, summary_rows)
                if use_ddp: dist.barrier()

            if is_main_process(use_ddp, local_rank):
                if best_fold_metrics is None: best_fold_metrics = evaluate(model, val_loader, device, use_ddp=False)
                save_done_marker(paths["done"], args.variant, fold_id, best_fold_f1, best_fold_metrics, args.epochs)
                print(f"Fold {fold_id} Best Results: {format_metrics(best_fold_metrics)}")
                fold_results.append(best_fold_metrics)

        if is_main_process(use_ddp, local_rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            save_summary_csv(summary_rows, epochs_csv)
            if not fold_results: return
            final_rows = []
            for metric_name in ["AUC", "AUPR", "Accuracy", "F1", "Precision", "Recall", "MCC", "Specificity"]:
                vals = [res[metric_name] for res in fold_results]
                mean_v, std_v = float(np.mean(vals)) * 100, float(np.std(vals)) * 100
                final_rows.append({"variant": args.variant, "metric": metric_name, "mean": mean_v, "std": std_v})
            save_summary_csv(final_rows, output_dir / f"{args.dataset_name}_{args.variant}_summary.csv")
            print(f"Saved summaries to: {output_dir.resolve()}")

    finally:

        pass

def run_experiment(args):

    if args.attention_layers:
        layers_to_run = args.attention_layers
    elif args.attention_layer is not None:
        layers_to_run = [args.attention_layer]
    else:
        layers_to_run = [config.contact_head_layer]

    base_output_dir = Path(args.output_dir)
    if len(set(layers_to_run)) != len(layers_to_run):
        raise ValueError(f"Duplicate attention layer(s) supplied: {layers_to_run}")
    if any(layer < 0 for layer in layers_to_run):
        raise ValueError("Attention layer indices must be non-negative (zero-based).")

    use_ddp = args.ddp or is_distributed_run()
    device, local_rank = setup_device(use_ddp)
    is_primary = is_main_process(use_ddp, local_rank)

    try:
        for layer in layers_to_run:
            run_args = argparse.Namespace(**vars(args))
            run_args.attention_layer = int(layer)
            run_args.attention_layers = None
            run_args.output_dir = str(base_output_dir / f"attention_layer_{layer}")
            if is_primary:
                print("\n" + "#" * 88)
                print(f"Starting attention-layer experiment: index={layer} | output={run_args.output_dir}")
                print("#" * 88)
            _run_single_experiment(run_args, use_ddp, device, local_rank)

        if is_primary:
            comparison_rows = []
            for layer in layers_to_run:
                layer_dir = base_output_dir / f"attention_layer_{layer}"
                summary_path = layer_dir / f"{args.dataset_name}_{args.variant}_summary.csv"
                if not summary_path.exists():
                    print(f"[warning] Summary not found for attention layer {layer}: {summary_path}")
                    continue
                layer_summary = pd.read_csv(summary_path)
                layer_summary.insert(0, "attention_layer_index", int(layer))
                comparison_rows.extend(layer_summary.to_dict("records"))
            if comparison_rows:
                base_output_dir.mkdir(parents=True, exist_ok=True)
                comparison_path = base_output_dir / "attention_layer_comparison_summary.csv"
                pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
                print(f"Saved layer comparison summary: {comparison_path.resolve()}")
    finally:
        cleanup_ddp(use_ddp)

def parse_args():
    parser = argparse.ArgumentParser(description="Yeast Hyperparameter Sensitivity Analysis.")
    parser.add_argument("--variant", type=str, default="full", choices=available_variants())
    parser.add_argument("--dataset-name", type=str, default="yeast")
    parser.add_argument("--data-path", type=str, default=str(_ROOT / "data/process_yeast/data_all.csv"))
    parser.add_argument("--fold-dir", type=str, default=str(_ROOT / "data/process_yeast/5_fold"))
    parser.add_argument("--split-source", type=str, default="kfold", choices=["fixed", "kfold", "single"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--only-fold", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="yeast_attention_layer_sensitivity")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ddp", action="store_true")

    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--attention-layer", type=int, default=None,
        help="Run one zero-based ESM-2 attention-layer index (e.g. 29)."
    )
    layer_group.add_argument(
        "--attention-layers", type=int, nargs="+", default=None,
        help="Run several zero-based ESM-2 layer indices sequentially (e.g. 25 27 29 31 32)."
    )

    parser.add_argument("--knn-k", type=int, default=None, help="Override config.knn_k (keep 10 for this experiment)")
    parser.add_argument("--contact-threshold", type=float, default=None, help="Override config.contact_threshold")
    parser.add_argument("--gat-heads", type=int, default=None, help="Override config.gat_heads")

    return parser.parse_args()

if __name__ == "__main__":
    run_experiment(parse_args())
