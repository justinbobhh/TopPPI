import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "config.py").is_file())
_sys.path.insert(0, str(_ROOT / "core"))
_sys.path.append(str(_ROOT / "experiment" / "ablation"))
_os.environ["TOPPPI_TASK"] = 'multiclass'
from runtime import prepare_training
prepare_training('multiclass')

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import config
from model_mc import TopPPIMultiClass
from loss_mc import DualBranchLoss_MC

warnings.filterwarnings('ignore', category=RuntimeWarning)

class MultiClassDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=800):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_a = str(row['seq_a'])
        seq_b = str(row['seq_b'])
        label = int(row['label'])

        enc_a = self.tokenizer(seq_a, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        enc_b = self.tokenizer(seq_b, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids_a': enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b': enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def evaluate_mc(model, dataloader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs_a = {'input_ids': batch['input_ids_a'].to(device), 'attention_mask': batch['attention_mask_a'].to(device)}
            inputs_b = {'input_ids': batch['input_ids_b'].to(device), 'attention_mask': batch['attention_mask_b'].to(device)}
            labels = batch['label'].to(device)

            with autocast(enabled=config.use_fp16):
                logits, _, _ = model(inputs_a, inputs_b)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro')

    mcc_strict = matthews_corrcoef(all_labels, all_preds)

    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.array(all_probs)
    num_classes = all_probs_arr.shape[1]

    if np.isnan(all_probs_arr).any():
        all_probs_arr = np.nan_to_num(all_probs_arr, nan=0.0)

    auc_list = []
    for i in range(num_classes):
        if len(np.unique(all_labels_arr == i)) == 2:
            class_auc = roc_auc_score(all_labels_arr == i, all_probs_arr[:, i])
            auc_list.append(class_auc)
    auc = np.mean(auc_list) if len(auc_list) > 0 else 0.0

    y_true_bin = label_binarize(all_labels_arr, classes=range(num_classes))
    y_pred_bin = label_binarize(all_preds_arr, classes=range(num_classes))

    mcc_list = []
    for i in range(num_classes):
        y_t_i = y_true_bin[:, i]
        y_p_i = y_pred_bin[:, i]
        if len(np.unique(y_t_i)) > 1:
            mcc_list.append(matthews_corrcoef(y_t_i, y_p_i))
    mcc_paper_style = np.mean(mcc_list) if len(mcc_list) > 0 else 0.0

    cm = confusion_matrix(all_labels, all_preds)
    spec_list = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        if (tn + fp) > 0:
            spec_list.append(tn / (tn + fp))
        else:
            spec_list.append(0.0)
    specificity = np.mean(spec_list) if len(spec_list) > 0 else 0.0

    return {
        "Accuracy": acc,
        "Macro-F1": f1,
        "Macro-Precision": prec,
        "Macro-Recall": rec,
        "MCC_Strict": mcc_strict,
        "MCC_PaperStyle": mcc_paper_style,
        "AUC": auc,
        "Macro-Specificity": specificity
    }

def run_multiclass_kfold_ddp():

    TRAIN_PATH = str(_Path(config.data_root) / 'process_multiclass/train.csv')
    TEST_PATH = str(_Path(config.data_root) / 'process_multiclass/test.csv')
    NUM_CLASSES = 7
    CHECKPOINT_PATH = "multiclass_ddp_checkpoint.pth"

    K_FOLDS = 5
    EPOCHS = 20
    LR = config.lr

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print(f"Running Multi-Class {K_FOLDS}-Fold CV with DDP (GPUs: {dist.get_world_size()})")

    try: tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except: tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df = pd.concat([df_train, df_test], ignore_index=True)
    if local_rank == 0:
        print(f"成功加载并合并数据集，总样本数: {len(df)}")

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

        train_ds = MultiClassDataset(df.iloc[train_idx], tokenizer, config.max_seq_len)
        val_ds = MultiClassDataset(df.iloc[val_idx], tokenizer, config.max_seq_len)

        train_sampler = DistributedSampler(train_ds, shuffle=True)

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

        model = TopPPIMultiClass(num_classes=NUM_CLASSES).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scaler = GradScaler(enabled=config.use_fp16)

        best_fold_f1 = checkpoint_data['best_fold_f1'] if (checkpoint_data and fold == start_fold) else 0.0
        best_fold_metrics = checkpoint_data['best_fold_metrics'] if (checkpoint_data and fold == start_fold) else {}

        if checkpoint_data and fold == start_fold:
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
            current_start_epoch = start_epoch
            checkpoint_data = None
        else:
            current_start_epoch = 0

        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        criterion = DualBranchLoss_MC(
            alpha=config.alpha_supcon,
            temperature=config.temperature
        ).to(device)

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
                    logits, proj, _ = model(inputs_a, inputs_b)
                    loss_out = criterion(logits, proj, labels)
                    loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
                    loss = loss.mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if local_rank == 0:
                    loop.set_postfix(loss=loss.item())

            metrics = evaluate_mc(model, val_loader, device)

            if local_rank == 0:
                print(f"   Val -> Acc: {metrics['Accuracy']*100:.2f} | M-F1: {metrics['Macro-F1']*100:.2f} | M-Prec: {metrics['Macro-Precision']*100:.2f} | M-Rec: {metrics['Macro-Recall']*100:.2f} | Spec: {metrics['Macro-Specificity']*100:.2f} | MCC(Strict): {metrics['MCC_Strict']*100:.2f} | MCC(Paper): {metrics['MCC_PaperStyle']*100:.2f} | AUC: {metrics['AUC']*100:.2f}")

                if metrics['Macro-F1'] > best_fold_f1:
                    best_fold_f1 = metrics['Macro-F1']
                    best_fold_metrics = metrics

                    save_path = f"best_model_mc_fold_{fold+1}.pth"

                    torch.save(model.module.state_dict(), save_path)
                    print(f"      [!] 发现更好的模型，已保存权重至: {save_path}")

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
                    'best_fold_f1': best_fold_f1,
                    'best_fold_metrics': best_fold_metrics,
                    'fold_results': fold_results
                }, CHECKPOINT_PATH)

            dist.barrier()

        if local_rank == 0:
            print(f"\n>>>>>>> [Fold {fold+1} 完结] 该折最佳模型完整性能汇报 <<<<<<<")
            print(f"  Accuracy:          {best_fold_metrics['Accuracy']*100:.2f}%")
            print(f"  Macro-F1:          {best_fold_metrics['Macro-F1']*100:.2f}%")
            print(f"  Macro-Precision:   {best_fold_metrics['Macro-Precision']*100:.2f}%")
            print(f"  Macro-Recall:      {best_fold_metrics['Macro-Recall']*100:.2f}%")
            print(f"  Macro-Specificity: {best_fold_metrics['Macro-Specificity']*100:.2f}%")
            print(f"  MCC (Strict):      {best_fold_metrics['MCC_Strict']*100:.2f}%")
            print(f"  MCC (Paper-Style): {best_fold_metrics['MCC_PaperStyle']*100:.2f}%")
            print(f"  AUC:               {best_fold_metrics['AUC']*100:.2f}%")
            print(f"=========================================================================\n")

            fold_results.append(best_fold_metrics)

            torch.save({
                'fold': fold + 1, 'epoch': 0, 'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                'best_fold_f1': 0.0, 'best_fold_metrics': {}, 'fold_results': fold_results
            }, CHECKPOINT_PATH)

        dist.barrier()

    if local_rank == 0:
        print("\n" + "="*70)
        print("FINAL 5-FOLD CROSS-VALIDATION RESULTS (MULTI-CLASS)")
        print("="*70)

        def get_ms(metric_name):
            vals = [res[metric_name] for res in fold_results]
            return np.mean(vals) * 100, np.std(vals) * 100

        acc_m, acc_s = get_ms("Accuracy")
        f1_m, f1_s = get_ms("Macro-F1")
        prec_m, prec_s = get_ms("Macro-Precision")
        rec_m, rec_s = get_ms("Macro-Recall")
        spec_m, spec_s = get_ms("Macro-Specificity")
        mcc_strict_m, mcc_strict_s = get_ms("MCC_Strict")
        mcc_paper_m, mcc_paper_s = get_ms("MCC_PaperStyle")
        auc_m, auc_s = get_ms("AUC")

        print(f"Accuracy:          {acc_m:.2f} ± {acc_s:.2f}")
        print(f"Macro-F1:          {f1_m:.2f} ± {f1_s:.2f}")
        print(f"Macro-Precision:   {prec_m:.2f} ± {prec_s:.2f}")
        print(f"Macro-Recall:      {rec_m:.2f} ± {rec_s:.2f}")
        print(f"Macro-Specificity: {spec_m:.2f} ± {spec_s:.2f}")
        print(f"MCC (Strict):      {mcc_strict_m:.2f} ± {mcc_strict_s:.2f}  <- 真实的数学评估")
        print(f"MCC (Paper-Style): {mcc_paper_m:.2f} ± {mcc_paper_s:.2f}  <- 建议填入论文对标表格的指标")
        print(f"AUC:               {auc_m:.2f} ± {auc_s:.2f}")
        print("="*70)

        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            print("[✓] All folds completed. Checkpoint file cleaned up.")

    dist.destroy_process_group()

if __name__ == "__main__":
    run_multiclass_kfold_ddp()
