import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'core'))
from config import Config

PATTERNS = {'yeast': 'best_model_fold_{fold}.pth', 'multispecies': 'multi_best_model_fold_{fold}.pth',
            'multiclass': 'best_model_mc_fold_{fold}.pth', 'human_virus': 'best_model_VF{fold}.pt'}

def arguments():
    p = argparse.ArgumentParser(description='TopPPI: inference.')
    p.add_argument('--task', required=True, choices=PATTERNS)
    p.add_argument('--input', type=Path, help='Custom CSV with seq_a,seq_b; IDs and labels are optional. Omit for paper validation folds.')
    p.add_argument('--fold', default=None, choices=('1','2','3','4','5','all'), help='Default: the author-selected fold for this task.')
    p.add_argument('--split', default='C2Virus', choices=('Train','Test','C1Virus','C2Virus'), help='Human-virus benchmark split only.')
    p.add_argument('--checkpoint', type=Path, help='Optional weight override for one fold.')
    p.add_argument('--esm-path', type=Path)
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--amp', choices=('off','float16'), default='off', help='Use float16 for the primary benchmark historical GPU evaluation.')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--limit', type=int, help='First N selected rows, for a smoke run; not a full benchmark.')
    p.add_argument('--show-config', action='store_true', help='Show paths/settings without ML imports.')
    p.add_argument('--check-only', action='store_true', help='Verify/reconstruct checkpoints and architecture dimensions without inference or ML model imports.')
    a = p.parse_args()
    model_manifest = json.loads((ROOT/'checkpoints/manifest.json').read_text(encoding='utf8'))
    if a.fold is None:
        a.fold = str(model_manifest['published_folds'][a.task])
    if a.fold == 'all':
        available = {int(row['fold']) for row in model_manifest['checkpoints'] if row['task'] == a.task}
        if available != set(range(1,6)):
            p.error('This release includes one selected fold per task. For other folds, supply --fold N --checkpoint PATH or retrain the five folds.')
    if a.batch_size < 1 or (a.limit is not None and a.limit < 1): p.error('Batch size and limit must be positive.')
    if not 0 <= a.threshold <= 1: p.error('Threshold must lie in [0,1].')
    if a.fold == 'all' and a.checkpoint: p.error('--checkpoint requires a single --fold.')
    return a

def paths(a, fold):
    checkpoint = a.checkpoint or ROOT/'checkpoints'/a.task/PATTERNS[a.task].format(fold=fold)
    return checkpoint.expanduser().resolve()

def input_frame(a, fold):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    if a.input:
        df = pd.read_csv(a.input)
        source = [str(a.input.resolve())]
        indices = np.arange(len(df))
    elif a.task == 'human_virus':
        path = a.data_root/f'human_virus/VF{fold}/VirusAppear_VF{fold}_{a.split}.csv'
        df = pd.read_csv(path)
        source = [str(path.resolve())]
        indices = np.arange(len(df))
    else:
        names = {'yeast': ['process_yeast/data_all.csv'],
                 'multispecies': ['process_multispecies/train_40.csv','process_multispecies/test_40.csv'],
                 'multiclass': ['process_multiclass/train.csv','process_multiclass/test.csv']}[a.task]
        source = [str((a.data_root/n).resolve()) for n in names]
        df = pd.concat([pd.read_csv(p) for p in source], ignore_index=True)
        indices = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))[fold-1][1]
    if a.limit is not None: indices = indices[:a.limit]
    frame = df.iloc[indices].copy()
    frame['_row_index'] = indices
    for name in ('seq_a','seq_b'):
        if name not in frame or frame[name].isna().any() or frame[name].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing or empty {name} in the input.')
    if frame.empty: raise ValueError('No input records.')
    if 'label' in frame and frame['label'].isna().any():
        raise ValueError('Either provide a label for every record or omit the label column.')
    return frame, source

def metrics(labels, probabilities, predictions, multiclass):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
    result = {'Accuracy': float(accuracy_score(labels,predictions)), 'MCC': float(matthews_corrcoef(labels,predictions))}
    average = 'weighted' if multiclass else 'binary'
    for name, fn in [('Precision',precision_score),('Recall',recall_score),('F1',f1_score)]:
        result[name] = float(fn(labels,predictions,average=average,zero_division=0))
    try:
        result['AUROC'] = float(roc_auc_score(labels,probabilities,average='weighted',multi_class='ovr')) if multiclass else float(roc_auc_score(labels,probabilities))
    except ValueError:
        result['AUROC'] = None
    if multiclass:
        result['Macro-F1'] = float(f1_score(labels,predictions,average='macro',zero_division=0))
        try: result['Macro-AUROC'] = float(roc_auc_score(labels,probabilities,average='macro',multi_class='ovr'))
        except ValueError: result['Macro-AUROC'] = None
    else:
        result['AP'] = float(average_precision_score(labels,probabilities)) if np.any(labels==1) else None
        tn,fp,fn,tp = confusion_matrix(labels,predictions,labels=[0,1]).ravel()
        result['Specificity'] = float(tn/(tn+fp)) if tn+fp else None
    return result

def main():
    a = arguments()
    cfg = Config(a.task)
    if a.esm_path: cfg.model_name = str(a.esm_path.expanduser().resolve())
    folds = list(range(1,6)) if a.fold == 'all' else [int(a.fold)]
    if a.show_config:
        print(json.dumps({'task': a.task, 'configuration': vars(cfg), 'checkpoints': [str(paths(a,f)) for f in folds],
                          'input': str(a.input) if a.input else 'manuscript validation records', 'amp': a.amp}, indent=2))
        return
    from checkpoints import read_checkpoint_state
    import torch
    if a.check_only:
        for fold in folds:
            state = read_checkpoint_state(paths(a,fold), expected_task=a.task)
            width = int(state['pool_v.attn_linear.weight'].shape[1])
            classes = int(state['classifier.4.weight'].shape[0])
            if width != cfg.gat_hidden_dim or classes != (7 if a.task=='multiclass' else 1):
                raise ValueError('Checkpoint dimensions do not match the selected task.')
            print(json.dumps({'task': a.task,'fold':fold,'tensors':len(state),'gat_width':width,'outputs':classes,'verified':True}))
        return
    import gc
    import numpy as np
    from transformers import AutoTokenizer
    from config import config
    config.select_task(a.task)
    config.model_name = cfg.model_name
    config.gradient_checkpointing = False
    device = torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable; choose --device cpu or use a CUDA server.')
    if a.amp=='float16' and device.type!='cuda': raise ValueError('--amp float16 requires CUDA.')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name,local_files_only=True)
    output = a.output_dir or ROOT/'outputs/inference'/a.task
    output.mkdir(parents=True,exist_ok=True)
    all_metrics = []
    for fold in folds:
        frame, sources = input_frame(a,fold)
        if a.task=='multiclass':
            from model_mc import TopPPIMultiClass
            model = TopPPIMultiClass(num_classes=7)
        else:
            from model import TopPPI
            model = TopPPI()
        state = read_checkpoint_state(paths(a,fold), expected_task=a.task)
        model.load_state_dict(state,strict=True)
        del state
        model.to(device).eval()
        batches = []
        with torch.inference_mode():
            for start in range(0,len(frame),a.batch_size):
                rows = frame.iloc[start:start+a.batch_size]
                inputs = []
                for col in ('seq_a','seq_b'):
                    tokens = tokenizer(rows[col].astype(str).str.replace(r'\s+','',regex=True).tolist(),padding='max_length',truncation=True,max_length=cfg.max_seq_len,return_tensors='pt')
                    inputs.append({k:tokens[k].to(device) for k in ('input_ids','attention_mask')})
                with torch.cuda.amp.autocast(enabled=a.amp=='float16'):
                    logits,_,_ = model(*inputs)
                probs = torch.softmax(logits.float(),dim=1) if a.task=='multiclass' else torch.sigmoid(logits).view(-1).float()
                if not torch.isfinite(probs).all(): raise ValueError('Non-finite prediction encountered.')
                batches.append(probs.cpu().numpy())
                print(f'fold {fold}: {min(start+a.batch_size,len(frame))}/{len(frame)}',flush=True)
        probabilities = np.concatenate(batches)
        predictions = probabilities.argmax(1) if a.task=='multiclass' else (probabilities>a.threshold).astype(int)
        destination = output/f'fold_{fold}_predictions.csv'
        probability_columns = [f'probability_{i}' for i in range(7)] if a.task=='multiclass' else ['probability']
        with destination.open('w',encoding='utf8',newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['row_index','protein_a_id','protein_b_id','label','prediction',*probability_columns])
            for pos,(_,row) in enumerate(frame.iterrows()):
                values = probabilities[pos].tolist() if a.task=='multiclass' else [float(probabilities[pos])]
                writer.writerow([int(row['_row_index']),row.get('protein_a_id',''),row.get('protein_b_id',''),row.get('label',''),int(predictions[pos]),*values])
        scores = {}
        if 'label' in frame:
            labels = frame['label'].to_numpy()
            if not np.isin(labels,range(7) if a.task=='multiclass' else (0,1)).all(): raise ValueError('Unexpected label values.')
            scores = metrics(labels,probabilities,predictions,a.task=='multiclass')
            all_metrics.append(scores)
        metadata = {'task':a.task,'fold':fold,'sources':sources,'checkpoint':str(paths(a,fold)), 'rows':len(frame),
                    'limit':a.limit,'input_kind':'custom' if a.input else 'selected validation / historical human-virus split',
                    'threshold':a.threshold,'amp':a.amp,'configuration':vars(config),'metrics':scores}
        (output/f'fold_{fold}_metrics.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
        print(json.dumps(scores,indent=2))
        del model
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    if len(all_metrics)>1:
        summary={}
        for name in all_metrics[0]:
            values=[m[name] for m in all_metrics if m[name] is not None]
            summary[name]={'mean':float(np.mean(values)),'population_std':float(np.std(values,ddof=0)),'folds':len(values)} if values else None
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

if __name__=='__main__':
    main()
