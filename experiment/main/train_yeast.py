import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "config.py").is_file())
_sys.path.insert(0, str(_ROOT / "core"))
_sys.path.append(str(_ROOT / "experiment" / "ablation"))
_os.environ["TOPPPI_TASK"] = 'yeast'
from runtime import prepare_training
prepare_training('yeast')

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
import random

from config import config
from model import TopPPI
from loss import DualBranchLoss

class YeastDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=800):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_a = row.get('seq_a') or row.get('protein_a') or row.get('sequence_a')
        seq_b = row.get('seq_b') or row.get('protein_b') or row.get('sequence_b')
        label = float(row['label'])

        enc_a = self.tokenizer(str(seq_a), padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        enc_b = self.tokenizer(str(seq_b), padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids_a': enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b': enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
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
        auc = roc_auc_score(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs)
    except:
        auc, aupr = 0.5, 0.0

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds)
    mcc = matthews_corrcoef(all_labels, preds)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": auc, "AUPR": aupr, "Accuracy": acc,
        "F1": f1, "Precision": prec, "Recall": rec,
        "MCC": mcc, "Specificity": specificity
    }

def run_yeast_kfold():
    DATA_PATH = str(_Path(config.data_root) / 'process_yeast/data_all.csv')
    K_FOLDS = 5
    EPOCHS = 25
    LR = config.lr

    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found: {DATA_PATH}")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(f"Running {K_FOLDS}-Fold CV on Yeast Dataset with DDP (GPUs: {dist.get_world_size()})")

    try: tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except: tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    df = pd.read_csv(DATA_PATH)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        if local_rank == 0:
            print(f"\n{'='*20} Fold {fold+1}/{K_FOLDS} {'='*20}")

        train_ds = YeastDataset(df.iloc[train_idx], tokenizer, config.max_seq_len)
        val_ds = YeastDataset(df.iloc[val_idx], tokenizer, config.max_seq_len)

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

        model = TopPPI().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR)

        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        model._set_static_graph()

        criterion = DualBranchLoss(
            alpha=config.alpha_supcon,
            beta=config.beta_focal,
            temperature=config.temperature,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        ).to(device)

        scaler = GradScaler(enabled=config.use_fp16)

        best_fold_f1 = 0
        best_fold_metrics = {}

        for epoch in range(EPOCHS):

            train_sampler.set_epoch(epoch)

            model.train()

            if local_rank == 0:
                loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
            else:
                loop = train_loader

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
                print(f"   Val -> AUC: {metrics['AUC']*100:.2f} | AUPR: {metrics['AUPR']*100:.2f} | Acc: {metrics['Accuracy']*100:.2f} | F1: {metrics['F1']*100:.2f} | Prec: {metrics['Precision']*100:.2f} | Rec: {metrics['Recall']*100:.2f} | MCC: {metrics['MCC']*100:.2f} | Spec: {metrics['Specificity']*100:.2f}")

                if metrics['F1'] > best_fold_f1:
                    best_fold_f1 = metrics['F1']
                    best_fold_metrics = metrics

                    save_path = f"best_model_fold_{fold+1}.pth"

                    torch.save(model.module.state_dict(), save_path)
                    print(f"      [!] 发现更好的模型，已保存权重至: {save_path}")

        if local_rank == 0:
            print(f"Fold {fold+1} Best Results (Saved): F1={best_fold_f1*100:.2f}%")
            fold_results.append(best_fold_metrics)

        dist.barrier()

    if local_rank == 0:
        print("\n" + "="*70)
        print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
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

    dist.destroy_process_group()

if __name__ == "__main__":
    run_yeast_kfold()
