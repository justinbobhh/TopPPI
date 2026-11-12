import os as _os
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "core" / "config.py").is_file())
_sys.path.insert(0, str(_ROOT / "core"))
_sys.path.append(str(_ROOT / "experiment" / "ablation"))
_os.environ["TOPPPI_TASK"] = 'multiclass'

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)

from config import config
from model_mc import TopPPIMultiClass

class MultiClassDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc_a = self.tokenizer(str(row['seq_a']), padding='max_length',
                               truncation=True, max_length=self.max_len, return_tensors='pt')
        enc_b = self.tokenizer(str(row['seq_b']), padding='max_length',
                               truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids_a': enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b': enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
            'label': torch.tensor(int(row['label']), dtype=torch.long),
        }

def compute_multiclass_metrics(y_true, y_pred, y_prob, num_classes=7):

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    labels = np.arange(num_classes)
    if y_prob.ndim != 2 or y_prob.shape != (len(y_true), num_classes):
        raise ValueError(f"Expected probabilities of shape (N, {num_classes}); got {y_prob.shape}")
    if not np.isfinite(y_prob).all():
        raise ValueError("NaN/Inf in predicted probabilities: inspect model evaluation; do not replace invalid values with zero.")
    row_sums = y_prob.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0, atol=1e-4):
        largest_error_idx = int(np.argmax(np.abs(row_sums - 1.0)))
        raise ValueError(
            'Multiclass probabilities are not normalized to 1 per example. '
            f'Max |sum-1|={abs(row_sums[largest_error_idx]-1.0):.8g} '
            f'at validation index {largest_error_idx}; '
            f'row sum={row_sums[largest_error_idx]:.8g}. '
            'FP32 Softmax should prevent ordinary FP16 rounding errors. '
            'Inspect logits and model configuration instead of silently renormalizing.'
        )
    if np.any((y_true < 0) | (y_true >= num_classes)):
        raise ValueError("True labels outside 0..num_classes-1.")
    if len(np.unique(y_true)) < num_classes:
        raise ValueError("Validation split does not contain every class; weighted OvR AUC is undefined for this fold.")

    result = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Weighted-Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted-Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted-F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'Weighted-AUC': roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr', labels=labels),

        'Macro-Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro-Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro-F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro-AUC': roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr', labels=labels),
        'MCC_Strict': matthews_corrcoef(y_true, y_pred),
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    n = cm.sum()
    specificity_values = []
    macro_ovr_mcc_values = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = n - tp - fp - fn
        specificity_values.append(tn / (tn + fp) if tn + fp else np.nan)
        truth_c = (y_true == c).astype(int)
        pred_c = (y_pred == c).astype(int)
        macro_ovr_mcc_values.append(matthews_corrcoef(truth_c, pred_c))
    result['Macro-Specificity'] = float(np.nanmean(specificity_values))
    result['MCC_MacroOvR'] = float(np.mean(macro_ovr_mcc_values))

    legacy_spec_den = cm[0, 0] + cm[0, 1]
    result['SSPPI-Legacy-Specificity'] = float(cm[0, 0] / legacy_spec_den) if legacy_spec_den else float('nan')
    legacy_numerator = cm[0, 0] * cm[1, 1] - cm[0, 1] * cm[1, 0]
    legacy_denominator = np.sqrt(
        (cm[1, 1] + cm[0, 1]) * (cm[1, 1] + cm[1, 0])
        * (cm[0, 0] + cm[0, 1]) * (cm[0, 0] + cm[1, 0])
    )
    result['SSPPI-Legacy-MCC'] = float(legacy_numerator / legacy_denominator) if legacy_denominator else float('nan')
    return result

def read_checkpoint_state(checkpoint_path):

    from checkpoints import read_checkpoint_state as read_complete_state
    return read_complete_state(checkpoint_path)

def infer_checkpoint_gat_hidden_dim(state_dict, checkpoint_path):

    keys = ['gat_adapter_v.norm.weight', 'gat_adapter_h.norm.weight',
            'pool_v.attn_linear.weight', 'pool_h.attn_linear.weight']
    missing = [k for k in keys if k not in state_dict]
    if missing:
        raise ValueError(
            f'Cannot infer architecture of {checkpoint_path}: missing {missing}. '
            'Please use the exact model implementation and configuration used in training.'
        )
    widths = [int(state_dict['gat_adapter_v.norm.weight'].numel()),
              int(state_dict['gat_adapter_h.norm.weight'].numel()),
              int(state_dict['pool_v.attn_linear.weight'].shape[1]),
              int(state_dict['pool_h.attn_linear.weight'].shape[1])]
    if len(set(widths)) != 1 or widths[0] <= 0:
        raise ValueError(f'Inconsistent hidden dimensions in {checkpoint_path}: {dict(zip(keys,widths))}')
    width = widths[0]
    if 'classifier.0.weight' in state_dict and int(state_dict['classifier.0.weight'].shape[1]) != 4 * width:
        raise ValueError(
            f'Fused classifier input does not match 4 x GAT width in {checkpoint_path}. '
            'This checkpoint may come from a different model version.'
        )
    return width

def load_model_weights(model, state_dict, checkpoint_path):

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f'Checkpoint architecture is still inconsistent with local model code: '
            f'{checkpoint_path}. Width was inferred automatically, but other '
            'model definitions or config parameters may differ from training. '
            'Please compare the actual training-time model_mc.py, layers.py, '
            'and config.py; do not use strict=False to hide shape errors.\\n'
            + str(exc)
        ) from exc

def evaluate_checkpoint(model, loader, device, use_fp16):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            a = {'input_ids': batch['input_ids_a'].to(device),
                 'attention_mask': batch['attention_mask_a'].to(device)}
            b = {'input_ids': batch['input_ids_b'].to(device),
                 'attention_mask': batch['attention_mask_b'].to(device)}

            with autocast(enabled=(use_fp16 and device.type == 'cuda')):
                logits, _, _ = model(a, b)

            if not torch.isfinite(logits).all():
                bad_indices = (~torch.isfinite(logits)).nonzero(as_tuple=False)[:5].tolist()
                raise RuntimeError(
                    f'Non-finite logits at validation batch {batch_idx}, positions {bad_indices}. '
                    'This is not a probability-normalization issue. Check the saved model, '
                    'input sequences, FP16 stability, and training-time configuration.'
                )
            probs_tensor = torch.softmax(logits.float(), dim=1)
            if not torch.isfinite(probs_tensor).all():
                raise RuntimeError(f'Non-finite FP32 Softmax values at validation batch {batch_idx}.')
            probs = probs_tensor.detach().cpu().numpy()
            y_true.extend(batch['label'].cpu().numpy().tolist())
            y_pred.extend(np.argmax(probs, axis=1).tolist())
            y_prob.extend(probs.tolist())
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path,'rb') as fh:
        for block in iter(lambda: fh.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def run(args):
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_path, test_path = Path(args.train_csv), Path(args.test_csv)
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f'Both original CSV files are required: {train_path}; {test_path}')
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], ignore_index=True)
    for key in ('seq_a','seq_b','label'):
        if key not in df.columns:
            raise ValueError(f'Missing expected CSV column {key!r}')
    if df[['seq_a','seq_b','label']].isna().any().any():
        raise ValueError('Input CSV has missing data. Use the *same* original processed train/test CSVs as training.')

    tokenizer_name = args.tokenizer if args.tokenizer else config.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        if args.tokenizer:
            raise
        tokenizer_name = 'facebook/esm2_t33_650M_UR50D'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    selected = list(range(1, 6)) if args.fold == 'all' else [int(args.fold)]
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    max_len = args.max_len if args.max_len else int(config.max_seq_len)
    records = []
    checkpoint_gat_widths = {}
    for fold in selected:
        val_idx = folds[fold-1][1]
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.checkpoint_pattern.format(fold=fold))
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing fold {fold} checkpoint: {checkpoint}. Use --checkpoint-pattern or --checkpoint.')
        print(f'Fold {fold}: loading {checkpoint} | validation records: {len(val_idx)}',flush=True)
        loader = DataLoader(
            MultiClassDataset(df.iloc[val_idx], tokenizer, max_len=max_len),
            batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        state_dict = read_checkpoint_state(checkpoint)
        trained_gat_width = infer_checkpoint_gat_hidden_dim(state_dict, checkpoint)
        checkpoint_gat_widths[str(fold)] = trained_gat_width
        if int(config.gat_hidden_dim) != trained_gat_width:
            print(
                f'  Architecture mismatch: config.gat_hidden_dim={config.gat_hidden_dim}, '
                f'checkpoint uses {trained_gat_width}. '
                f'Using checkpoint width {trained_gat_width} FOR EVALUATION ONLY.',
                flush=True,
            )
            config.gat_hidden_dim = trained_gat_width
        else:
            print(f'  Checkpoint GAT hidden dimension: {trained_gat_width}', flush=True)
        model = TopPPIMultiClass(num_classes=7).to(device)
        load_model_weights(model,state_dict,checkpoint)
        del state_dict
        y_true,y_pred,y_prob = evaluate_checkpoint(model,loader,device,use_fp16=bool(config.use_fp16))
        metrics = compute_multiclass_metrics(y_true,y_pred,y_prob,num_classes=7)
        metrics['Fold'] = fold
        records.append(metrics)
        print(f"  Accuracy={metrics['Accuracy']*100:.3f}% | W-F1={metrics['Weighted-F1']*100:.3f}% | W-AUC={metrics['Weighted-AUC']*100:.3f}% | Macro-F1={metrics['Macro-F1']*100:.3f}%", flush=True)
        if args.save_predictions:
            pred_df = pd.DataFrame({'row_index_in_pooled_csv': val_idx,
                                    'y_true': y_true,'y_pred': y_pred})
            for cls in range(y_prob.shape[1]):
                pred_df[f'prob_class_{cls}'] = y_prob[:,cls]
            pred_df.to_csv(root / f'predictions_fold_{fold}.csv', index=False)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out_df = pd.DataFrame(records).set_index('Fold').sort_index()
    out_df.to_csv(root / 'per_fold_metrics.csv')
    if len(selected) == 5:
        summary = pd.DataFrame({
            'metric': out_df.columns,
            'mean_percent': [out_df[c].mean()*100 for c in out_df.columns],
            'std_percent_ddof0': [out_df[c].std(ddof=0)*100 for c in out_df.columns]
        })
        summary.to_csv(root / 'metrics_mean_std.csv',index=False)
        print('\nFive-fold mean ± population std (percentage points):')
        for _, row in summary.iterrows():
            print(f"  {row['metric']:28s}: {row['mean_percent']:.3f} ± {row['std_percent_ddof0']:.3f}")
    else:
        print('Only one fold was evaluated; no five-fold mean/std file is produced.')

    protocol = {
        'train_csv': str(train_path), 'test_csv': str(test_path),
        'sha256_train_csv': file_sha256(train_path),
        'sha256_test_csv': file_sha256(test_path),
        'n_train_source_rows':len(train_df),
        'n_test_source_rows':len(test_df),
        'n_pooled_rows':len(df),
        'split':'KFold(n_splits=5, shuffle=True, random_state=42) on pooled rows',
        'evaluated_folds': selected,
        'tokenizer': tokenizer_name, 'max_seq_len':max_len,
        'checkpoint_selection':'original training selected best epoch by Macro-F1',
        'checkpoint_gat_hidden_dims': checkpoint_gat_widths,
        'architecture_note':'GAT width inferred from saved weights; other shape-free hyperparameters must match original training settings',
        'main_ssppi_metrics':'weighted precision, weighted recall, weighted F1, weighted OvR AUROC, accuracy',
        'ssppi_legacy_specificity_mcc_warning':'Original SSPPI public code applies binary metrics to the top-left 2x2 of 7x7 confusion matrix; nonstandard for seven classes',
        'evaluation_status':'checkpoint-selected validation fold, not an untouched test set',
    }
    (root/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'\nSaved outputs to {root.resolve()}')

def build_parser():
    parser = argparse.ArgumentParser(description='TopPPI: evaluate multiclass.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-csv', default=str(_ROOT / 'data/process_multiclass/train.csv'))
    parser.add_argument('--test-csv', default=str(_ROOT / 'data/process_multiclass/test.csv'))
    parser.add_argument('--fold', default='3', choices=['all','1','2','3','4','5'])
    parser.add_argument('--checkpoint', default=None,help='Checkpoint path for a single --fold only')
    parser.add_argument('--checkpoint-pattern', default=str(_ROOT / 'checkpoints/multiclass/best_model_mc_fold_{fold}.pth'),help='Pattern for all-fold evaluation; fold is 1..5')
    parser.add_argument('--tokenizer', default=None,help='Default: config.model_name')
    parser.add_argument('--max-len', type=int, default=None, help='Default: config.max_seq_len')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output-dir', default=str(_ROOT / 'outputs/multiclass_evaluation'))
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    if args.checkpoint and args.fold == 'all':
        parser.error('--checkpoint requires specifying a single --fold (1..5); for multiple folds use --checkpoint-pattern')
    run(args)
