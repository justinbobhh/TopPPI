import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "config.py").is_file())
_sys.path.insert(0, str(_ROOT / "core"))
_sys.path.append(str(_ROOT / "experiment" / "ablation"))
_os.environ["TOPPPI_TASK"] = 'multispecies'
from runtime import prepare_training
prepare_training('multispecies')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
import pandas as pd
import os
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix
)
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from config import config
from model import TopPPI
from loss import DualBranchLoss

class MultiSpeciesDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=1024):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc_a = self.tokenizer(str(row['seq_a']), padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        enc_b = self.tokenizer(str(row['seq_b']), padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids_a': enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b': enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
            'label': torch.tensor(float(row['label']), dtype=torch.float)
        }

def evaluate(model, dataloader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs_a = {'input_ids': batch['input_ids_a'].to(device), 'attention_mask': batch['attention_mask_a'].to(device)}
            inputs_b = {'input_ids': batch['input_ids_b'].to(device), 'attention_mask': batch['attention_mask_b'].to(device)}
            labels = batch['label'].to(device)

            with autocast(enabled=config.use_fp16):
                logits, _, _ = model(inputs_a, inputs_b)

            probs = torch.sigmoid(logits).squeeze().cpu().float().numpy()
            if probs.ndim == 0: probs = np.expand_dims(probs, axis=0)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        specificity = 0.0

    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = 0.5
    try: aupr = average_precision_score(all_labels, all_probs)
    except: aupr = 0.0

    return {
        "AUC": auc, "AUPR": aupr, "Accuracy": accuracy_score(all_labels, preds),
        "F1": f1_score(all_labels, preds), "Precision": precision_score(all_labels, preds, zero_division=0),
        "Recall": recall_score(all_labels, preds), "MCC": matthews_corrcoef(all_labels, preds),
        "Specificity": specificity
    }

def run_multispecies_kfold_merged_ddp():
    TRAIN_PATH = str(_Path(config.data_root) / 'process_multispecies/train_40.csv')
    TEST_PATH = str(_Path(config.data_root) / 'process_multispecies/test_40.csv')
    CHECKPOINT_PATH = "multi_checkpoint.pth"

    K_FOLDS = 5
    EPOCHS = 12
    LR = 5e-5

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print(f"Error: 找不到文件 {TRAIN_PATH} 或 {TEST_PATH}")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df = pd.concat([df_train, df_test], ignore_index=True)

    if local_rank == 0:
        print(f"Merged Dataset Size: {len(df)} samples")
        print(f"Running {K_FOLDS}-Fold CV with DDP & Checkpointing (GPUs: {dist.get_world_size()})")

    try: tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except: tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    start_fold = 0
    start_epoch = 0
    fold_results = []
    checkpoint_data = None

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint_data = torch.load(CHECKPOINT_PATH, map_location='cpu')
        start_fold = checkpoint_data['fold']
        start_epoch = checkpoint_data['epoch']
        fold_results = checkpoint_data['fold_results']
        if local_rank == 0:
            print(f"\n[!] 发现 Checkpoint！正在从 Fold {start_fold + 1}, Epoch {start_epoch + 1} 恢复训练...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        if fold < start_fold:
            continue

        if local_rank == 0:
            print(f"\n{'='*20} Fold {fold+1}/{K_FOLDS} {'='*20}")

        train_ds = MultiSpeciesDataset(df.iloc[train_idx], tokenizer, config.max_seq_len)
        val_ds = MultiSpeciesDataset(df.iloc[val_idx], tokenizer, config.max_seq_len)

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

        model = TopPPI().to(device)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        scaler = GradScaler(enabled=config.use_fp16)

        best_fold_auc = checkpoint_data['best_fold_auc'] if (checkpoint_data and fold == start_fold) else 0.0
        best_fold_metrics = checkpoint_data['best_fold_metrics'] if (checkpoint_data and fold == start_fold) else {}

        if checkpoint_data and fold == start_fold:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
            checkpoint_data = None

        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        model._set_static_graph()

        n_pos = np.sum(df.iloc[train_idx]['label'].values == 1)
        total_train = len(train_idx)
        focal_alpha_val = (total_train - n_pos) / total_train

        criterion = DualBranchLoss(
            alpha=config.alpha_supcon, beta=config.beta_focal,
            temperature=config.temperature, focal_alpha=focal_alpha_val, focal_gamma=config.focal_gamma
        ).to(device)

        current_start_epoch = start_epoch if fold == start_fold else 0

        for epoch in range(current_start_epoch, EPOCHS):
            train_sampler.set_epoch(epoch)
            model.train()

            if local_rank == 0: loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
            else: loop = train_loader

            for batch in loop:
                inputs_a = {'input_ids': batch['input_ids_a'].to(device), 'attention_mask': batch['attention_mask_a'].to(device)}
                inputs_b = {'input_ids': batch['input_ids_b'].to(device), 'attention_mask': batch['attention_mask_b'].to(device)}
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                with autocast(enabled=config.use_fp16):
                    logits, proj, extra_scores = model(inputs_a, inputs_b)
                    loss_out = criterion(logits, proj, labels)
                    loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out

                    dummy_loss = (proj * 0).sum() + (extra_scores[0] * 0).sum() + (extra_scores[1] * 0).sum()
                    loss = loss + dummy_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if local_rank == 0:
                    loop.set_postfix(loss=loss.item())

            metrics = evaluate(model, val_loader, device)

            if local_rank == 0:
                print(
                    f"   Val -> AUC: {metrics['AUC']*100:.2f} | AUPR: {metrics['AUPR']*100:.2f} | "
                    f"Acc: {metrics['Accuracy']*100:.2f} | F1: {metrics['F1']*100:.2f} | "
                    f"Prec: {metrics['Precision']*100:.2f} | Rec: {metrics['Recall']*100:.2f} | "
                    f"MCC: {metrics['MCC']*100:.2f} | Spec: {metrics['Specificity']*100:.2f}"
                )

                if metrics['AUC'] > best_fold_auc:
                    old_auc = best_fold_auc
                    best_fold_auc = metrics['AUC']
                    best_fold_metrics = metrics
                    torch.save(model.module.state_dict(), f"multi_best_model_fold_{fold+1}.pth")
                    print(f"      >>> [新突破] Epoch {epoch+1} 性能提升! AUC: {old_auc*100:.2f}% -> {best_fold_auc*100:.2f}%. 权重已存盘。")

                next_epoch = epoch + 1
                next_fold = fold
                if next_epoch == EPOCHS:
                    next_fold = fold + 1
                    next_epoch = 0

                torch.save({
                    'fold': next_fold,
                    'epoch': next_epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_fold_auc': best_fold_auc,
                    'best_fold_metrics': best_fold_metrics,
                    'fold_results': fold_results
                }, CHECKPOINT_PATH)

            dist.barrier()

        if local_rank == 0:
            print(f"\n>>>>>>> [Fold {fold+1} 完结] 该折最佳模型（Best Model）完整性能汇报 <<<<<<<")
            print(f"  AUC:       {best_fold_metrics['AUC']*100:.2f}%  |  AUPR:      {best_fold_metrics['AUPR']*100:.2f}%")
            print(f"  Accuracy:  {best_fold_metrics['Accuracy']*100:.2f}%  |  F1:        {best_fold_metrics['F1']*100:.2f}%")
            print(f"  Precision: {best_fold_metrics['Precision']*100:.2f}%  |  Recall:    {best_fold_metrics['Recall']*100:.2f}%")
            print(f"  MCC:       {best_fold_metrics['MCC']*100:.2f}%  |  Spec:      {best_fold_metrics['Specificity']*100:.2f}%")
            print(f"=========================================================================\n")

            fold_results.append(best_fold_metrics)

            torch.save({
                'fold': fold + 1,
                'epoch': 0,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_fold_auc': 0.0,
                'best_fold_metrics': {},
                'fold_results': fold_results
            }, CHECKPOINT_PATH)

        dist.barrier()

    if local_rank == 0:
        print("\n" + "="*70)
        print("FINAL 5-FOLD CROSS-VALIDATION RESULTS (Merged train_40 + test_40)")
        print("="*70)

        def get_ms(metric_name):
            vals = [res[metric_name] for res in fold_results]
            return np.mean(vals) * 100, np.std(vals) * 100

        auc_m, auc_s = get_ms("AUC")
        aupr_m, aupr_s = get_ms("AUPR")
        acc_m, acc_s = get_ms("Accuracy")
        f1_m, f1_s = get_ms("F1")
        prec_m, prec_s = get_ms("Precision")
        rec_m, rec_s = get_ms("Recall")
        mcc_m, mcc_s = get_ms("MCC")
        spec_m, spec_s = get_ms("Specificity")

        print(f"AUC:       {auc_m:.2f} ± {auc_s:.2f}  |  AUPR:      {aupr_m:.2f} ± {aupr_s:.2f}")
        print(f"Accuracy:  {acc_m:.2f} ± {acc_s:.2f}  |  F1:        {f1_m:.2f} ± {f1_s:.2f}")
        print(f"Precision: {prec_m:.2f} ± {prec_s:.2f}  |  Recall:    {rec_m:.2f} ± {rec_s:.2f}")
        print(f"MCC:       {mcc_m:.2f} ± {mcc_s:.2f}  |  Spec:      {spec_m:.2f} ± {spec_s:.2f}")
        print("="*70)

        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            print("[✓] All folds completed. Checkpoint file cleaned up.")

    dist.destroy_process_group()

if __name__ == "__main__":
    run_multispecies_kfold_merged_ddp()
