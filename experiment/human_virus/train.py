import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "config.py").is_file())
_sys.path.insert(0, str(_ROOT / "core"))
_sys.path.append(str(_ROOT / "experiment" / "ablation"))
_os.environ["TOPPPI_TASK"] = 'human_virus'
from runtime import prepare_training
prepare_training('human_virus')

import os
import csv
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from tqdm import tqdm
from pathlib import Path
import datetime

from model import TopPPI
from config import config
from loss import DualBranchLoss

BASE_DIR = _Path(config.data_root)
CSV_ROOT = BASE_DIR / "human_virus"

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

class DeepGNHVDataset(Dataset):
    def __init__(self, csv_files):
        self.samples = []
        for csv_file in csv_files:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append({
                        'seq_v': row['seq_a'],
                        'seq_h': row['seq_b'],
                        'label': int(row['label'])
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        v_enc = tokenizer(s['seq_v'], truncation=True, max_length=config.max_seq_len,
                          padding='max_length', return_tensors='pt')
        h_enc = tokenizer(s['seq_h'], truncation=True, max_length=config.max_seq_len,
                          padding='max_length', return_tensors='pt')
        return {
            'virus_input_ids': v_enc['input_ids'].squeeze(0),
            'virus_attn_mask': v_enc['attention_mask'].squeeze(0),
            'host_input_ids': h_enc['input_ids'].squeeze(0),
            'host_attn_mask': h_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(s['label'], dtype=torch.float32)
        }

def collate_fn(batch):
    return {
        'virus_inputs': {
            'input_ids': torch.stack([b['virus_input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['virus_attn_mask'] for b in batch])
        },
        'host_inputs': {
            'input_ids': torch.stack([b['host_input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['host_attn_mask'] for b in batch])
        },
        'labels': torch.stack([b['label'] for b in batch])
    }

def get_train_files_for_test_vf(test_vf_name):
    train_csv = CSV_ROOT / test_vf_name / f"VirusAppear_{test_vf_name}_Train.csv"
    if train_csv.exists():
        return [train_csv]
    else:
        raise FileNotFoundError(f"未找到训练文件：{train_csv}")

def get_test_file_for_vf(test_vf_name):
    test_csv = CSV_ROOT / test_vf_name / f"VirusAppear_{test_vf_name}_C2Virus.csv"
    if test_csv.exists():
        return test_csv
    else:
        raise FileNotFoundError(f"未找到测试文件：{test_csv}")

def compute_all_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        'AUROC': roc_auc_score(labels, probs),
        'AUPRC': average_precision_score(labels, probs),
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds, zero_division=0),
        'Recall': recall_score(labels, preds, zero_division=0),
        'F1': f1_score(labels, preds, zero_division=0),
        'MCC': matthews_corrcoef(labels, preds)
    }

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            v_inp = {k: v.to(device) for k, v in batch['virus_inputs'].items()}
            h_inp = {k: v.to(device) for k, v in batch['host_inputs'].items()}
            labels = batch['labels'].to(device)

            logits, _, _ = model(v_inp, h_inp)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    return compute_all_metrics(np.array(all_labels), np.array(all_probs))

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, local_rank):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", disable=(local_rank not in [-1, 0]))
    for batch in pbar:
        v_inp = {k: v.to(device) for k, v in batch['virus_inputs'].items()}
        h_inp = {k: v.to(device) for k, v in batch['host_inputs'].items()}
        labels = batch['labels'].to(device)

        if scaler:
            with torch.cuda.amp.autocast():
                logits, proj, _ = model(v_inp, h_inp)
                loss, _, _ = criterion(logits.view(-1), proj, labels)

                dummy_loss = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        dummy_loss = dummy_loss + p.sum() * 0.0
                loss = loss + dummy_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits, proj, _ = model(v_inp, h_inp)
            loss, _, _ = criterion(logits.view(-1), proj, labels)

            dummy_loss = 0.0
            for p in model.parameters():
                if p.requires_grad:
                    dummy_loss = dummy_loss + p.sum() * 0.0
            loss = loss + dummy_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def print_metrics_table(metrics_dict, title="Performance Metrics"):
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"{'Metric':<15} {'Value':>10} {'Std':>10}")
    print(f"{'-'*35}")
    for metric, (mean, std) in metrics_dict.items():
        print(f"{metric:<15} {mean:>10.4f} {std:>10.4f}")
    print(f"{'='*50}\n")

def main():

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=4))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        local_rank = 0

    if local_rank == 0:
        vf_list = [d.name for d in sorted(CSV_ROOT.iterdir()) if d.is_dir() and d.name.startswith('VF')]
        print(f"发现 VF 文件夹: {vf_list}")

    vf_list = [d.name for d in sorted(CSV_ROOT.iterdir()) if d.is_dir() and d.name.startswith('VF')]

    all_results = {metric: [] for metric in ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']}

    for test_vf in vf_list:
        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"开始交叉验证实验: {test_vf} (当前为测试集，其余已在当前Train.csv内)")
            print(f"{'='*60}")

        model = TopPPI().to(device)
        if is_distributed:

            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            model._set_static_graph()

        criterion = DualBranchLoss(alpha=config.alpha_supcon, beta=config.beta_focal,
                                   focal_alpha=config.focal_alpha, focal_gamma=config.focal_gamma)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scaler = torch.cuda.amp.GradScaler() if config.use_fp16 else None

        train_files = get_train_files_for_test_vf(test_vf)
        test_file = get_test_file_for_vf(test_vf)

        train_dataset = DeepGNHVDataset(train_files)
        test_dataset = DeepGNHVDataset([test_file])

        train_sampler = DistributedSampler(train_dataset) if is_distributed else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        if local_rank == 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )
            print(f"训练集样本总数: {len(train_dataset)} | 测试集样本数: {len(test_dataset)}")

        best_metrics = None
        best_auprc = 0.0

        for epoch in range(EPOCHS):

            if is_distributed:
                train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, local_rank)

            if local_rank == 0:

                torch.cuda.empty_cache()

                eval_model = model.module if hasattr(model, "module") else model
                metrics = evaluate(eval_model, test_loader, device)
                print(f"Epoch {epoch+1}/{EPOCHS} | Rank0 Loss: {train_loss:.4f} | AUROC: {metrics['AUROC']:.4f} | AUPRC: {metrics['AUPRC']:.4f}")

                if metrics['AUPRC'] > best_auprc:
                    best_auprc = metrics['AUPRC']
                    best_metrics = metrics
                    torch.save(eval_model.state_dict(), f"best_model_{test_vf}.pt")

            if is_distributed:
                dist.barrier()

        if local_rank == 0 and best_metrics is not None:
            for metric in all_results:
                all_results[metric].append(best_metrics[metric])

            print(f"\n{test_vf} 最佳结果 (按AUPRC保存):")
            for metric, value in best_metrics.items():
                print(f"  {metric}: {value:.4f}")

    if local_rank == 0:
        final_metrics = {}
        for metric, values in all_results.items():
            final_metrics[metric] = (np.mean(values), np.std(values))

        print_metrics_table(final_metrics, "你的模型在 C2Virus 场景下的最终性能 (5-Fold 平均)")

        deepgnhv_metrics = {
            'AUROC': (0.825, 0.002), 'AUPRC': (0.573, 0.001), 'Accuracy': (0.750, 0.002),
            'Precision': (0.730, 0.003), 'Recall': (0.770, 0.002), 'F1': (0.745, 0.002),
            'MCC': (0.480, 0.003)
        }

        print("\n" + "="*60)
        print("对比总结 (你的模型 vs DeepGNHV)")
        print("="*60)
        print(f"{'Metric':<15} {'Your Model':>15} {'DeepGNHV':>15} {'Difference':>15}")
        print("-"*60)
        for metric in final_metrics:
            your_mean = final_metrics[metric][0]
            deep_mean = deepgnhv_metrics[metric][0]
            diff = your_mean - deep_mean
            diff_symbol = "▲" if diff > 0 else "▼" if diff < 0 else "="
            print(f"{metric:<15} {your_mean:>15.4f} {deep_mean:>15.4f} {diff_symbol} {abs(diff):>13.4f}")
        print("="*60)

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
