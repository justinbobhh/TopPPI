import argparse
import json
from pathlib import Path
import sys
import numpy as np
from common import HERE, DATASETS, read_json, write_json, sha256

def sample_rows(labels, limit, seed, mode='proportional'):
    labels=np.asarray(labels)
    if labels.ndim != 1 or not np.isfinite(labels).all() or not np.equal(labels, labels.astype(int)).all():
        raise ValueError('Labels must be finite integer class indices')
    if limit < 1:
        raise ValueError('max-points must be positive')
    if len(labels)<=limit:return np.arange(len(labels))
    classes,counts=np.unique(labels,return_counts=True)
    if limit < len(classes):raise ValueError('max-points must accommodate all classes')
    rng=np.random.default_rng(seed)
    if mode=='balanced':
        allocation=np.full(len(classes), min(counts.min(),limit//len(classes)),dtype=int)
    else:
        target=counts*limit/len(labels)
        allocation=np.maximum(1,np.floor(target).astype(int))
        while allocation.sum()>limit:
            eligible=np.where(allocation>1,allocation-target,-np.inf)
            allocation[np.argmax(eligible)]-=1
        while allocation.sum()<limit:
            eligible=np.where(allocation<counts,target-allocation,-np.inf)
            allocation[np.argmax(eligible)]+=1
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels==c),n,replace=False) for c,n in zip(classes,allocation)]))

def unpack_state(checkpoint):
    if not isinstance(checkpoint,dict):raise ValueError('Checkpoint must contain a state dictionary')
    for key in ('model_state','model_state_dict','state_dict'):
        if key in checkpoint:
            state=checkpoint[key];break
    else:state=checkpoint
    if not state or not all(hasattr(v,'shape') for v in state.values()):
        raise ValueError('No recognized model state dictionary in checkpoint')
    return {k.removeprefix('module.'):v for k,v in state.items()}

def load_weights(model, checkpoint):
    state=unpack_state(checkpoint)
    if checkpoint.get('checkpoint_mode') == 'trainable':
        result=model.load_state_dict(state,strict=False)
        missing=set(result.missing_keys)
        missing_trainable=[n for n,p in model.named_parameters() if p.requires_grad and n in missing]
        if result.unexpected_keys or missing_trainable:
            raise ValueError(f'Incomplete trainable checkpoint: {missing_trainable}; unexpected {result.unexpected_keys}')
    else:
        model.load_state_dict(state,strict=True)

def main():
    p=argparse.ArgumentParser(description='TopPPI: predict.')
    p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--csv',type=Path,nargs='+',required=True,help='Exact evaluation CSV, or original CSVs in training concat order with --fold')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-name',required=True,help='Matching local ESM-2 model directory')
    p.add_argument('--model-dir',type=Path,default=HERE.parents[2]/'core',help='Default authoritative core; change only for a matching historical model')
    p.add_argument('--config-json',type=Path,help='JSON object of original executed config values; paths overridden by --model-name')
    p.add_argument('--gat-width',type=int,help='Explicit historical width, e.g.128; never silently use512 for128 weights')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',required=True,help='e.g. validation-fold1, C2Virus-selected, or independent-test')
    p.add_argument('--selection-note',required=True,help='How the checkpoint was chosen; this text is retained in provenance')
    p.add_argument('--fold',type=int,help='Optional unstratified KFold reconstruction: folds1..5, split seed42 by default')
    p.add_argument('--k-folds',type=int,default=5)
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--rows-npy',type=Path,help='Alternative exact evaluation row indices in concatenated CSVs')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--max-points',type=int,default=2000)
    p.add_argument('--sampling',choices=['proportional','balanced'],default='proportional')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.output.suffix!='.npz':p.error('output must end in .npz')
    if a.output.exists() or a.output.with_suffix('.json').exists():p.error('Output exists; use a new name')
    if a.batch_size<1:p.error('batch-size must be positive')
    if a.fold is not None and a.rows_npy is not None:p.error('Choose --fold OR --rows-npy')
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    tables=[]
    for path in a.csv:
        df=pd.read_csv(path)
        if not {'seq_a','seq_b','label'}<=set(df):raise ValueError(f'Missing seq_a/seq_b/label in {path}')
        if df[['seq_a','seq_b','label']].isna().any().any():raise ValueError('Missing sequences/labels: supply the exact processed evaluation data')
        tables.append(df)
    frame=pd.concat(tables,ignore_index=True)
    all_labels=frame['label'].to_numpy()
    classes=7 if a.dataset=='multiclass' else 2
    if not np.isfinite(all_labels).all() or not np.equal(all_labels,all_labels.astype(int)).all() or not np.isin(all_labels,np.arange(classes)).all():
        raise ValueError(f'{a.dataset} requires integer labels0..{classes-1}')
    rows=np.arange(len(frame))
    if a.fold is not None:
        from sklearn.model_selection import KFold
        if not 1<=a.fold<=a.k_folds:raise ValueError('fold out of range')
        rows=list(KFold(a.k_folds,shuffle=True,random_state=a.split_seed).split(frame))[a.fold-1][1]
    elif a.rows_npy:
        rows=np.load(a.rows_npy,allow_pickle=False)
        if rows.dtype.kind not in 'iu' or rows.ndim!=1 or len(set(rows))!=len(rows) or (rows<0).any() or (rows>=len(frame)).any():
            raise ValueError('Invalid evaluation row indices')
    labels=all_labels[rows].astype(np.int64)
    selected=sample_rows(labels,a.max_points,a.seed,a.sampling)
    chosen_rows=rows[selected]
    selected_frame=frame.iloc[chosen_rows]
    sys.path.insert(0,str(a.model_dir.resolve()))
    from config import config
    if a.config_json:
        overrides=read_json(a.config_json)
        for key,value in overrides.items():
            if not hasattr(config,key):raise ValueError(f'Unknown configuration field: {key}')
            setattr(config,key,value)
    config.model_name=a.model_name
    config.gradient_checkpointing=False
    if a.gat_width is not None:config.gat_hidden_dim=a.gat_width
    if a.dataset=='multiclass':
        from model_mc import TopPPIMultiClass
        model=TopPPIMultiClass(num_classes=7)
    else:
        from model import TopPPI
        model=TopPPI()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if isinstance(checkpoint,dict) and checkpoint.get('format')=='topppi-shared-backbone-v1':
        from checkpoints import read_checkpoint_state
        checkpoint={'state_dict':read_checkpoint_state(a.checkpoint)}
    load_weights(model,checkpoint)
    del checkpoint
    device=torch.device(a.device)
    model.to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(a.model_name)
    captured=[]
    handle=model.classifier.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach().float().cpu().numpy()))
    projections,probabilities=[],[]
    try:
        with torch.inference_mode():
            for start in range(0,len(selected_frame),a.batch_size):
                batch=selected_frame.iloc[start:start+a.batch_size]
                inputs=[]
                for col in ('seq_a','seq_b'):
                    encoded=tokenizer(batch[col].astype(str).tolist(),padding='max_length',truncation=True,max_length=config.max_seq_len,return_tensors='pt')
                    inputs.append({k:v.to(device) for k,v in encoded.items()})
                logits,projected,_=model(*inputs)
                projections.append(projected.float().cpu().numpy())
                probability=torch.softmax(logits,dim=1) if classes==7 else torch.sigmoid(logits).reshape(-1)
                probabilities.append(probability.float().cpu().numpy())
                print(f'{a.dataset}: {min(start+a.batch_size,len(selected_frame))}/{len(selected_frame)}',flush=True)
    finally:handle.remove()
    x=np.concatenate(captured); projection=np.concatenate(projections)
    if len(x)!=len(chosen_rows) or not np.isfinite(x).all() or not np.isfinite(projection).all():
        raise ValueError('Invalid captured features')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    metadata={'dataset':a.dataset,'split':a.split,'selection_note':a.selection_note,
        'checkpoint_sha256':sha256(a.checkpoint),'source_csv_sha256':[sha256(path) for path in a.csv],
        'config':vars(config),'code_sha256':{f.name:sha256(f) for f in sorted(a.model_dir.glob('*.py'))},
        'n_evaluation':len(rows),'n_exported':len(chosen_rows),'sampling':a.sampling,'seed':a.seed,
        'evaluation_class_counts':{str(c):int((labels==c).sum()) for c in range(classes)},
        'fold':a.fold,'k_folds':a.k_folds,'split_seed':a.split_seed,
        'note':'One checkpoint, observed labels. No correctness filtering. Inference only; not newly trained results.'}
    np.savez_compressed(a.output,classifier_input=x,projection=projection,labels=all_labels[chosen_rows].astype(np.int64),
                        row_index=chosen_rows,probability=np.concatenate(probabilities),metadata=json.dumps(metadata))
    write_json(a.output.with_suffix('.json'),metadata)
    print(f'Saved {a.output}')

if __name__=='__main__':main()
