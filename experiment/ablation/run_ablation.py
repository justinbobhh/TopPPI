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
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from config import config
from loss import DualBranchLoss
from model_ablation import (
    ABLATION_VARIANTS,
    available_variants,
    build_ablation_model,
    resolve_model_name,
)

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

        enc_a = self.tokenizer(
            str(seq_a),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            str(seq_b),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }

def is_distributed_run():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def setup_distributed(force_ddp=False):
    use_ddp = force_ddp or is_distributed_run()

    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested, but CUDA is not available.")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return use_ddp, device, local_rank, rank, world_size

def cleanup_distributed(use_ddp):
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def compute_metrics(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs > 0.5).astype(int)

    try:
        auc = roc_auc_score(labels, probs)
        aupr = average_precision_score(labels, probs)
    except ValueError:
        auc, aupr = 0.5, 0.0

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": float(auc),
        "AUPR": float(aupr),
        "Accuracy": float(acc),
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "MCC": float(mcc),
        "Specificity": float(specificity),
    }

def evaluate_rank0(model, dataloader, device, use_ddp, rank):

    if use_ddp and rank != 0:
        return None

    eval_model = model.module if use_ddp and hasattr(model, "module") else model
    eval_model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_a = {
                "input_ids": batch["input_ids_a"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask_a"].to(device, non_blocking=True),
            }
            inputs_b = {
                "input_ids": batch["input_ids_b"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask_b"].to(device, non_blocking=True),
            }
            labels = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=getattr(config, "use_fp16", True) and device.type == "cuda"):
                logits, _, _ = eval_model(inputs_a, inputs_b)

            probs = torch.sigmoid(logits).view(-1).detach().cpu().float().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    return compute_metrics(all_labels, all_probs)

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    use_ddp,
    rank,
    epoch,
):
    model.train()
    total_loss = 0.0
    total_steps = 0

    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)

    iterator = dataloader
    if is_main_process(rank):
        iterator = tqdm(dataloader, desc=f"Ep {epoch + 1}", dynamic_ncols=True)

    for batch in iterator:
        inputs_a = {
            "input_ids": batch["input_ids_a"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask_a"].to(device, non_blocking=True),
        }
        inputs_b = {
            "input_ids": batch["input_ids_b"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask_b"].to(device, non_blocking=True),
        }
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=getattr(config, "use_fp16", True) and device.type == "cuda"):
            logits, proj, extra_scores = model(inputs_a, inputs_b)
            loss_out = criterion(logits, proj, labels)
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out

            if use_ddp:
                loss = loss + (proj * 0).sum()
                loss = loss + (extra_scores[0] * 0).sum() + (extra_scores[1] * 0).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        total_steps += 1

        if is_main_process(rank):
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = total_loss / max(total_steps, 1)

    if use_ddp:
        loss_tensor = torch.tensor(mean_loss, dtype=torch.float32, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= dist.get_world_size()
        mean_loss = float(loss_tensor.item())

    return mean_loss

def format_metrics(metrics):
    return (
        f"AUC: {metrics['AUC'] * 100:.2f} | "
        f"AUPR: {metrics['AUPR'] * 100:.2f} | "
        f"Acc: {metrics['Accuracy'] * 100:.2f} | "
        f"F1: {metrics['F1'] * 100:.2f} | "
        f"Prec: {metrics['Precision'] * 100:.2f} | "
        f"Rec: {metrics['Recall'] * 100:.2f} | "
        f"MCC: {metrics['MCC'] * 100:.2f} | "
        f"Spec: {metrics['Specificity'] * 100:.2f}"
    )

def resolve_path(path_value, script_dir):
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        script_dir / path,
        script_dir.parent / path,
        script_dir.parent / "data" / path,
    ]
    return next((p for p in candidates if p.exists()), path)

def build_fold_pairs(args, script_dir, rank):
    if args.split_source == "fixed":
        fold_dir = resolve_path(args.fold_dir, script_dir)
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fixed fold directory not found: {fold_dir}")

        pairs = []
        for fold in range(1, args.k_folds + 1):
            train_file = fold_dir / f"train{fold}.csv"
            valid_file = fold_dir / f"valid{fold}.csv"
            if not train_file.exists() or not valid_file.exists():
                raise FileNotFoundError(
                    f"Missing fixed split files for fold {fold}: {train_file} / {valid_file}"
                )
            pairs.append((train_file, valid_file))

        if is_main_process(rank):
            print(f"Using fixed Yeast fold files from: {fold_dir}")
        return pairs

    data_path = resolve_path(args.data_path, script_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    pairs = [(df.iloc[tr].copy(), df.iloc[va].copy()) for tr, va in kf.split(df)]

    if is_main_process(rank):
        print(f"Using regenerated KFold split from: {data_path}")
    return pairs

def get_fold_dataframes(args, train_source, val_source):
    if args.split_source == "fixed":
        return pd.read_csv(train_source), pd.read_csv(val_source)
    return train_source, val_source

def checkpoint_paths(output_dir, variant, fold_id):
    output_dir = Path(output_dir)
    stem = f"yeast_{variant}_fold_{fold_id}"
    return {
        "best": output_dir / f"{stem}_best.pth",
        "last": output_dir / f"{stem}_last.pt",
        "done": output_dir / f"{stem}_done.json",
    }

def model_state_dict(model, use_ddp):
    return model.module.state_dict() if use_ddp else model.state_dict()

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    use_ddp,
    args,
    fold_id,
    epoch,
    best_fold_f1,
    best_fold_metrics,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": args.variant,
        "fold": fold_id,
        "epoch": epoch,
        "model_state_dict": model_state_dict(model, use_ddp),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_fold_f1": float(best_fold_f1),
        "best_fold_metrics": best_fold_metrics,
        "run_args": vars(args),
    }
    torch.save(payload, path)

def load_checkpoint(path, model, optimizer, scaler, device, use_ddp, args):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if ckpt.get("variant") != args.variant:
        raise RuntimeError(
            f"Checkpoint variant={ckpt.get('variant')} does not match requested variant={args.variant}."
        )

    target_model = model.module if use_ddp and hasattr(model, "module") else model
    target_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt

def save_done_marker(path, args, fold_id, best_fold_f1, best_fold_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": args.variant,
        "fold": fold_id,
        "best_fold_f1": float(best_fold_f1),
        "best_fold_metrics": best_fold_metrics,
        "completed_epochs": int(args.epochs),
        "seed": int(args.seed),
        "k_folds": int(args.k_folds),
        "split_source": args.split_source,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_done_marker(path):
    return json.loads(path.read_text(encoding="utf-8"))

def save_summary_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

def run_yeast_ablation(args):
    use_ddp, device, local_rank, rank, world_size = setup_distributed(args.ddp)

    set_seed(args.seed + rank)

    try:
        if is_main_process(rank):
            print("=" * 80)
            print(f"Variant      : {args.variant}")
            print(f"Description  : {ABLATION_VARIANTS[args.variant]['description']}")
            print(f"DDP          : {use_ddp}")
            print(f"World size   : {world_size}")
            print(f"Visible GPUs : {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            print(f"Epochs       : {args.epochs}")
            print(f"Folds        : {args.k_folds}")
            print(f"Split        : {args.split_source}")
            print(f"Seed         : {args.seed}")
            print(f"LR           : {args.lr}")
            print(f"Resume       : {args.resume}")
            print(f"Output       : {args.output_dir}")
            print("=" * 80)

        script_dir = Path(__file__).resolve().parent
        tokenizer = AutoTokenizer.from_pretrained(resolve_model_name(config.model_name))
        fold_pairs = build_fold_pairs(args, script_dir, rank)
        output_dir = Path(args.output_dir)

        fold_results = []
        epoch_rows = []

        for fold_index, (train_source, val_source) in enumerate(fold_pairs):
            fold_id = fold_index + 1
            if args.only_fold is not None and fold_id != args.only_fold:
                continue

            paths = checkpoint_paths(output_dir, args.variant, fold_id)

            if args.resume and paths["done"].exists():
                done_payload = load_done_marker(paths["done"])
                completed_epochs = int(done_payload.get("completed_epochs", 0))

                if completed_epochs > args.epochs:
                    raise RuntimeError(
                        f"Fold {fold_id} already has a completed {completed_epochs}-epoch run in {output_dir}, "
                        f"but this launch asks for {args.epochs} epochs. Use a NEW --output-dir for the "
                        f"{args.epochs}-epoch experiment so old and new results are not mixed."
                    )

                if completed_epochs == args.epochs:
                    if is_main_process(rank):
                        print(
                            f"[resume] Fold {fold_id} already completed {completed_epochs} epochs; skipping."
                        )
                        fold_results.append(done_payload["best_fold_metrics"])
                    if use_ddp:
                        dist.barrier()
                    continue

                if is_main_process(rank):
                    print(
                        f"[resume] Fold {fold_id} was completed only to {completed_epochs} epochs; "
                        f"continuing toward {args.epochs}."
                    )

            if is_main_process(rank):
                print(f"\n{'=' * 25} Fold {fold_id}/{args.k_folds} {'=' * 25}")

            train_df, val_df = get_fold_dataframes(args, train_source, val_source)
            train_ds = YeastDataset(train_df, tokenizer, config.max_seq_len)
            val_ds = YeastDataset(val_df, tokenizer, config.max_seq_len)

            train_sampler = None
            if use_ddp:
                train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=args.seed,
                    drop_last=False,
                )

            loader_kwargs = {
                "batch_size": config.batch_size,
                "num_workers": args.num_workers,
                "pin_memory": device.type == "cuda",
                "persistent_workers": args.num_workers > 0,
            }

            train_loader = DataLoader(
                train_ds,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                **loader_kwargs,
            )

            val_loader = None
            if is_main_process(rank):
                val_loader = DataLoader(
                    val_ds,
                    shuffle=False,
                    **loader_kwargs,
                )

            model = build_ablation_model(args.variant).to(device)
            optimizer = optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=args.lr,
            )

            if use_ddp:
                model = DDP(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,
                )

                if hasattr(model, "_set_static_graph"):
                    model._set_static_graph()

            criterion = DualBranchLoss(
                alpha=config.alpha_supcon,
                beta=config.beta_focal,
                temperature=config.temperature,
                focal_alpha=config.focal_alpha,
                focal_gamma=getattr(config, "focal_gamma", 2.0),
            ).to(device)

            scaler = GradScaler(enabled=getattr(config, "use_fp16", True) and device.type == "cuda")
            best_fold_f1 = -1.0
            best_fold_metrics = None
            start_epoch = 0

            if args.resume and paths["last"].exists():
                ckpt = load_checkpoint(
                    paths["last"], model, optimizer, scaler, device, use_ddp, args
                )
                start_epoch = int(ckpt["epoch"]) + 1
                best_fold_f1 = float(ckpt.get("best_fold_f1", -1.0))
                best_fold_metrics = ckpt.get("best_fold_metrics")

                if is_main_process(rank):
                    print(
                        f"[resume] Fold {fold_id}: epoch {start_epoch}/{args.epochs}, "
                        f"best F1={best_fold_f1 * 100:.2f}%"
                    )

            if start_epoch >= args.epochs:
                if is_main_process(rank):
                    print(
                        f"Fold {fold_id}: checkpoint already reached target epoch count ({args.epochs})."
                    )
                    if best_fold_metrics is not None:
                        fold_results.append(best_fold_metrics)
                        save_done_marker(
                            paths["done"], args, fold_id, best_fold_f1, best_fold_metrics
                        )
                if use_ddp:
                    dist.barrier()
                continue

            for epoch in range(start_epoch, args.epochs):
                train_loss = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    scaler=scaler,
                    device=device,
                    use_ddp=use_ddp,
                    rank=rank,
                    epoch=epoch,
                )

                metrics = evaluate_rank0(model, val_loader, device, use_ddp, rank)

                if is_main_process(rank):
                    print(
                        f"Fold {fold_id} | Epoch {epoch + 1:02d}/{args.epochs:02d} | "
                        f"Loss {train_loss:.4f} | {format_metrics(metrics)}"
                    )

                    epoch_rows.append(
                        {
                            "variant": args.variant,
                            "fold": fold_id,
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            **metrics,
                        }
                    )

                    if metrics["F1"] > best_fold_f1:
                        best_fold_f1 = metrics["F1"]
                        best_fold_metrics = metrics
                        output_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(model_state_dict(model, use_ddp), paths["best"])
                        print(
                            f"  [best] Fold {fold_id}: new best F1={best_fold_f1 * 100:.2f}% -> {paths['best']}"
                        )

                    save_checkpoint(
                        path=paths["last"],
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        use_ddp=use_ddp,
                        args=args,
                        fold_id=fold_id,
                        epoch=epoch,
                        best_fold_f1=best_fold_f1,
                        best_fold_metrics=best_fold_metrics,
                    )
                    print(f"  [ckpt] Saved resumable checkpoint -> {paths['last']}")

                if use_ddp:
                    dist.barrier()

            if is_main_process(rank):
                if best_fold_metrics is None:
                    raise RuntimeError(f"Fold {fold_id} produced no validation metrics.")

                save_done_marker(
                    paths["done"], args, fold_id, best_fold_f1, best_fold_metrics
                )
                fold_results.append(best_fold_metrics)
                print(f"Fold {fold_id} BEST -> {format_metrics(best_fold_metrics)}")

            if use_ddp:
                dist.barrier()

            del model, optimizer, criterion, scaler, train_loader
            if val_loader is not None:
                del val_loader
            torch.cuda.empty_cache()

        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            save_summary_csv(
                epoch_rows,
                output_dir / f"yeast_{args.variant}_epochs.csv",
            )

            expected = 1 if args.only_fold is not None else args.k_folds
            if len(fold_results) != expected:
                raise RuntimeError(
                    f"Only {len(fold_results)} completed fold result(s) were collected; expected {expected}. "
                    "Do not use this as the final ablation result."
                )

            if args.only_fold is not None:
                print("WARNING: --only-fold was used. This is a debug run, not a final 5-fold result.")

            print("\n" + "=" * 80)
            print(f"FINAL YEAST ABLATION RESULTS: {args.variant}")
            print("=" * 80)

            final_rows = []
            for metric_name in [
                "AUC",
                "AUPR",
                "Accuracy",
                "F1",
                "Precision",
                "Recall",
                "MCC",
                "Specificity",
            ]:
                vals = [x[metric_name] for x in fold_results]
                mean_v = float(np.mean(vals)) * 100
                std_v = float(np.std(vals)) * 100
                final_rows.append(
                    {
                        "variant": args.variant,
                        "metric": metric_name,
                        "mean": mean_v,
                        "std": std_v,
                    }
                )
                print(f"{metric_name:<12} {mean_v:>8.2f} +/- {std_v:.2f}")

            save_summary_csv(
                final_rows,
                output_dir / f"yeast_{args.variant}_summary.csv",
            )
            print("=" * 80)
            print(f"Results saved to: {output_dir.resolve()}")

    finally:
        cleanup_distributed(use_ddp)

def parse_args():
    parser = argparse.ArgumentParser(
        description="DDP + resumable Yeast ablation training for TopPPI."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="full",
        choices=available_variants(),
        help="Ablation variant to run.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(_ROOT / "data/process_yeast/data_all.csv"),
        help="Yeast CSV used when --split-source kfold.",
    )
    parser.add_argument(
        "--fold-dir",
        type=str,
        default=str(_ROOT / "data/process_yeast/5_fold"),
        help="Folder containing train1.csv...train5.csv and valid1.csv...valid5.csv.",
    )
    parser.add_argument(
        "--split-source",
        type=str,
        default="kfold",
        choices=["fixed", "kfold"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--only-fold", type=int, default=4)
    parser.add_argument("--lr", type=float, default=getattr(config, "lr", 1e-4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=str(_ROOT / "outputs/ablation"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each fold from yeast_<variant>_fold_<n>_last.pt and skip folds already completed to the requested epoch count.",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Force DDP. Normally torch.distributed.run/torchrun enables DDP automatically.",
    )
    parser.add_argument("--list-variants", action="store_true")
    args = parser.parse_args()

    if args.list_variants:
        for name, meta in ABLATION_VARIANTS.items():
            print(f"{name}: {meta['description']}")
        raise SystemExit(0)

    return args

if __name__ == "__main__":
    run_yeast_ablation(parse_args())
