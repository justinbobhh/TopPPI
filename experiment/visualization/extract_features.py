import argparse
import contextlib
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def read_path_list(root, override, default_paths, fallback=None, label='dataset'):
    if override:
        paths = [Path(p).expanduser() for p in override]
        paths = [p if p.is_absolute() else root / p for p in paths]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'{label} explicitly specified files missing: {missing}')
        return paths
    for candidate in default_paths:
        paths = [root / s for s in candidate]
        if all(p.is_file() for p in paths):
            return paths
    if fallback:
        paths = [root / s for s in fallback]
        existing = [p for p in paths if p.is_file()]
        if len(existing) == 1:
            return existing
        if len(existing) > 1:
            raise RuntimeError(f'{label}: multiple possible combined files {existing}; pass explicit --...-files')
    options = ' OR '.join(' + '.join(g) for g in default_paths)
    raise FileNotFoundError(f'{label} not found under {root}. Expected {options}. '
                            'Inspect the real CSV paths and use --yeast-files, --ms01-files, --ms10-files, --ms25-files, --ms40-files, or --msany-files.')

def load_data(paths, label):
    chunks = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(chunks, ignore_index=True)
    a = next((k for k in ('seq_a', 'protein_a', 'sequence_a') if k in df.columns), None)
    b = next((k for k in ('seq_b', 'protein_b', 'sequence_b') if k in df.columns), None)
    if not a or not b or 'label' not in df.columns:
        raise ValueError(f'{label}: required seq_a/seq_b/label (or supported sequence aliases), got {list(df.columns)}')
    if df[a].isna().any() or df[b].isna().any():
        raise ValueError(f'{label}: missing protein sequence; do not silently fill missing sequences')
    numeric_labels = pd.to_numeric(df['label'], errors='coerce')
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        raise ValueError(f'{label}: expected binary labels 0/1 only')
    df['label'] = numeric_labels.astype(int)

    return df, a, b

def pair_hash(sa, sb):

    x = ''.join(str(sa).split()).upper()
    y = ''.join(str(sb).split()).upper()
    if x > y: x, y = y, x
    digest = hashlib.sha256()
    digest.update(x.encode('utf8')); digest.update(b'\0'); digest.update(y.encode('utf8'))
    return digest.digest()

def keys_for(df, col_a, col_b):
    return [pair_hash(a,b) for a,b in zip(df[col_a].tolist(), df[col_b].tolist())]

def resolve_checkpoint(root, chosen, name):
    if chosen:
        p = Path(chosen).expanduser()
        p = p if p.is_absolute() else root / p
        if not p.is_file():
            raise FileNotFoundError(f'{name} model checkpoint not found: {p}')
        return p

    candidates = [root / name,
                  root / 'checkpoints' / name,
                  root / 'multispecies_results' / name,
                  root / 'yeast_results' / name]
    hits = [p for p in candidates if p.is_file()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f'Multiple {name} files found: {hits}. Pass explicit checkpoint path.')
    raise FileNotFoundError(f'Cannot find {name}. Search on server: find . -type f -name "{name}". '
                            'Then pass --yeast-ckpt or --ms-ckpt. Do not use a checkpoint from another dataset.')

def find_sources(args):
    root = args.root.resolve()

    yeast = read_path_list(root, args.yeast_files,
        [['data/process_yeast/data_all.csv']], label='Yeast')
    msbase = 'data/process_multispecies/'

    if args.ms40_source == 'merged':
        ms40 = read_path_list(root, args.ms40_files,
            [[msbase+'dataset40_merged.csv']], label='Multi-species_40')
    else:
        ms40 = read_path_list(root, args.ms40_files,
            [[msbase+'train_40.csv', msbase+'test_40.csv']], label='Multi-species_40')
    ms01 = read_path_list(root, args.ms01_files,
        [[msbase+'train_01.csv',msbase+'test_01.csv'],
         [msbase+'train_1.csv',msbase+'test_1.csv']],
        fallback=[msbase+'dataset01_merged.csv', msbase+'dataset1_merged.csv'],
        label='Multi-species_01')
    ms10 = read_path_list(root,args.ms10_files,
        [[msbase+'train_10.csv',msbase+'test_10.csv']],
        fallback=[msbase+'dataset10_merged.csv', msbase+'dataset_10_merged.csv'],
        label='Multi-species_10')
    ms25 = read_path_list(root,args.ms25_files,
        [[msbase+'train_25.csv',msbase+'test_25.csv']],
        fallback=[msbase+'dataset25_merged.csv', msbase+'dataset_25_merged.csv'],
        label='Multi-species_25')
    msany = read_path_list(root,args.msany_files,
        [[msbase+'train_any.csv',msbase+'test_any.csv']],
        fallback=[msbase+'datasetany_merged.csv', msbase+'dataset_any_merged.csv'],
        label='Multi-species_any')
    return {'Yeast':yeast, 'Multi-species_01':ms01,
            'Multi-species_10':ms10, 'Multi-species_25':ms25,
            'Multi-species_40':ms40, 'Multi-species_any':msany}

def prepare(args):
    root = args.root.resolve()
    print('[PROJECT ROOT]', root)
    print('[MULTI-SPECIES_40 DATA SOURCE]', args.ms40_source)
    source_files=find_sources(args)
    dfs, ab_cols, keys={},{},{}
    for name,paths in source_files.items():
        df,a,b=load_data(paths,name)
        dfs[name]=df;ab_cols[name]=(a,b);keys[name]=keys_for(df,a,b)
        print(f'[DATA] {name}: {len(df)} rows, +/- {int((df.label==1).sum())}/{int((df.label==0).sum())}')
        for p in paths: print('       ',p)

    check_merged = root / 'data/process_multispecies/dataset40_merged.csv'
    original_pair_files = [root / 'data/process_multispecies/train_40.csv',
                           root / 'data/process_multispecies/test_40.csv']
    if check_merged.is_file() and all(p.is_file() for p in original_pair_files):
        check_df = pd.read_csv(check_merged, low_memory=False)
        pair_df = pd.concat([pd.read_csv(p, low_memory=False) for p in original_pair_files],
                            ignore_index=True)
        cols = ['seq_a','seq_b','label']
        if all(c in check_df and c in pair_df for c in cols):
            exact_same_order = (len(check_df) == len(pair_df) and all(
                check_df[c].astype(str).to_numpy().tolist() == pair_df[c].astype(str).to_numpy().tolist()
                for c in cols))
            if exact_same_order:
                print('[DATA-CHECK] dataset40_merged.csv is row-identical to train_40.csv + test_40.csv '
                      'for seq_a, seq_b and label. Both reconstruct the SAME Fold 3.')
            else:
                print('[DATA-CHECK WARNING] dataset40_merged.csv and train_40.csv + test_40.csv '
                      'are NOT row-identical. You must use the precise source used when training '
                      'multi_best_model_fold_3.pth; default is train+test from the supplied training code.')
        else:
            print('[DATA-CHECK WARNING] Could not compare row order between merged and train+test '
                  '(missing seq_a/seq_b/label columns).')

    root_yeast = root / 'process_yeast/data_all.csv'
    screenshot_yeast = root / 'SSPPI/data/process_yeast/data_all.csv'
    if root_yeast.is_file() and screenshot_yeast.is_file():
        alternate = pd.read_csv(root_yeast, low_memory=False)
        selected = dfs['Yeast']
        cols = ['seq_a','seq_b','label']
        if all(c in alternate and c in selected for c in cols):
            same = (len(alternate) == len(selected) and all(
                alternate[c].astype(str).to_numpy().tolist() == selected[c].astype(str).to_numpy().tolist()
                for c in cols))
            if not same:
                msg = ('Yeast CSV at ./process_yeast/data_all.csv differs from '
                       './SSPPI/data/process_yeast/data_all.csv. Confirm which generated the Yeast checkpoint; '
                       'for the former use --yeast-files ./process_yeast/data_all.csv.')
                print('[DATA-CHECK WARNING]', msg)
                if args.mode == 'extract' and not args.yeast_files:
                    raise RuntimeError('Ambiguous Yeast training CSV; do not extract with an unverified Fold 3. ' + msg)
            else:
                print('[DATA-CHECK] Both Yeast CSV copies have the same row order and labels.')

    if len(source_files['Multi-species_40']) == 1:
        print('[FOLD-CHECK] Using dataset40_merged.csv. Fold 3 is correct ONLY IF this was '
              'the training CSV, in exactly the original row order.')
    else:
        print('[FOLD-CHECK] Using train_40.csv + test_40.csv in the order used by '
              'train_multispecies(1).py (the script that saves multi_best_model_fold_X.pth).')

    yeast_ckpt = resolve_checkpoint(root,args.yeast_ckpt,f'best_model_fold_{args.yeast_fold}.pth')
    ms_ckpt = resolve_checkpoint(root,args.ms_ckpt,f'multi_best_model_fold_{args.ms_fold}.pth')
    print('[MODEL] Yeast:',yeast_ckpt)
    print('[MODEL] Multi-species_40:',ms_ckpt)
    print(f'[FOLD] Yeast={args.yeast_fold}, Multi-species_40={args.ms_fold}, KFold random_state=42')

    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    ytr,yval=list(kf.split(dfs['Yeast']))[args.yeast_fold-1]
    mtr,mval=list(kf.split(dfs['Multi-species_40']))[args.ms_fold-1]
    train_sets={'Yeast':set(keys['Yeast'][i] for i in ytr),
                'Multi':set(keys['Multi-species_40'][i] for i in mtr)}
    indices={'Yeast':np.asarray(yval, dtype=int),
             'Multi-species_40':np.asarray(mval,dtype=int),
             'Multi-species_01':np.arange(len(dfs['Multi-species_01'])),
             'Multi-species_10':np.arange(len(dfs['Multi-species_10'])),
             'Multi-species_25':np.arange(len(dfs['Multi-species_25'])),
             'Multi-species_any':np.arange(len(dfs['Multi-species_any']))}
    stats={}
    for name in ['Yeast','Multi-species_01','Multi-species_10','Multi-species_25','Multi-species_40','Multi-species_any']:
        candidates=indices[name]
        training=train_sets['Yeast' if name=='Yeast' else 'Multi']
        seen=np.array([keys[name][i] in training for i in candidates],dtype=bool)
        unseen=candidates[~seen]
        stats[name]={'candidate_rows':int(len(candidates)),
                     'exact_pair_overlap_with_training':int(seen.sum()),
                     'eligible_unseen_pair_rows':int(len(unseen)),
                     'eligible_positive':int((dfs[name].iloc[unseen].label==1).sum()),
                     'eligible_negative':int((dfs[name].iloc[unseen].label==0).sum())}
        print(f'[AUDIT] {name}: candidate={len(candidates)}, overlaps model train={seen.sum()}, '
              f'eligible={len(unseen)}, positive={stats[name]["eligible_positive"]}, '
              f'negative={stats[name]["eligible_negative"]}')
        indices[name]=unseen
    print('[NOTICE] Exact sequence-pair overlap filtering is NOT a homology/CD-HIT identity check. '
          'Filtered panels are descriptive representation visualizations, not formal external generalization tests.')
    return root, source_files, dfs, ab_cols, indices, stats, yeast_ckpt,ms_ckpt

def sample_by_class(df, rows, k, seed):
    if k < 1: raise ValueError('--per-class must be positive')
    rng=np.random.default_rng(seed)
    chosen=[]
    for cls in (0,1):
        pool=np.asarray([i for i in rows if df.iloc[int(i)].label==cls],dtype=int)
        if not len(pool):
            raise ValueError(f'No class {cls} samples remain after exact-pair overlap filter.')
        chosen.extend(rng.choice(pool,min(k,len(pool)),replace=False).tolist())
    return np.sort(np.asarray(chosen,dtype=int))

def extract_features(args, prepared):
    import torch
    from torch.utils.data import Dataset,DataLoader
    from transformers import AutoTokenizer

    root,source_files,dfs,ab_cols,indices,stats,yckpt,mckpt = prepared
    snapshot=Path(__file__).resolve().parents[2] / 'core'
    if not all((snapshot/x).is_file() for x in ['model.py','layers.py','config.py']):
        raise FileNotFoundError('Missing core model files. Use the complete TopPPI repository.')
    sys.path.insert(0,str(snapshot))
    from config import config
    from model import TopPPI

    esm=Path(args.esm_path).expanduser().resolve()
    if not esm.is_dir(): raise FileNotFoundError(f'ESM-2 folder missing: {esm}')
    tok=AutoTokenizer.from_pretrained(str(esm))
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('[DEVICE]',device)

    class PairDataset(Dataset):
        def __init__(self,df,cols,rows):
            self.df=df; self.a,self.b=cols;self.rows=rows
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):
            rid=int(self.rows[index]);row=self.df.iloc[rid]
            a=tok(str(row[self.a]),max_length=800,padding='max_length',truncation=True,return_tensors='pt')
            b=tok(str(row[self.b]),max_length=800,padding='max_length',truncation=True,return_tensors='pt')
            return a['input_ids'][0],a['attention_mask'][0],b['input_ids'][0],b['attention_mask'][0],int(row.label),rid

    def state_dict_for(path):
        from checkpoints import read_checkpoint_state
        loaded=read_checkpoint_state(path)
        key='gat_adapter.norm.weight'
        if key not in loaded:raise ValueError(f'{path} does not contain {key}. Check architecture.')
        return loaded,int(loaded[key].numel())

    def create_model(path):
        state,width=state_dict_for(path)
        config.model_name=str(esm)
        config.max_seq_len=800
        config.gat_hidden_dim=width
        config.gradient_checkpointing=False
        config.use_fp16=False
        print(f'[MODEL] Constructing checkpoint width={width} for {path}')
        model=TopPPI().to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    out=Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    summary={}
    panel_groups=[('Yeast',yckpt),('Multi-species_01',mckpt),('Multi-species_10',mckpt),('Multi-species_25',mckpt),('Multi-species_40',mckpt),('Multi-species_any',mckpt)]
    previous_ckpt=None
    model=None
    for panel,ckpt in panel_groups:
        output=out/(panel.replace('-','_')+'.npz')
        if output.exists() and not args.overwrite:
            print('[SKIP] File already exists:',output,'(pass --overwrite to replace)')
            summary[panel]=dict(status='skipped',file=str(output))
            continue
        if previous_ckpt != ckpt:
            if model is not None:
                del model
                if device.type=='cuda':torch.cuda.empty_cache()
            model=create_model(ckpt)
            previous_ckpt=ckpt
        df=dfs[panel]
        rows=sample_by_class(df,indices[panel],args.per_class,args.seed)
        ds=PairDataset(df,ab_cols[panel],rows)
        dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
        captured=[]
        def hook(_module,ins):captured.append(ins[0].detach().float().cpu().numpy())
        handle=model.classifier[0].register_forward_pre_hook(hook)
        features=[];labels=[];row_indices=[]
        try:
            with torch.no_grad():
                for step,batch in enumerate(dl,1):
                    ids_a,mask_a,ids_b,mask_b,yy,rr=batch
                    aa={'input_ids':ids_a.to(device),'attention_mask':mask_a.to(device)}
                    bb={'input_ids':ids_b.to(device),'attention_mask':mask_b.to(device)}
                    captured.clear()
                    _=model(aa,bb)
                    if len(captured)!=1:raise RuntimeError('Cannot capture one fused representation from classifier input')
                    features.append(captured.pop()); labels.append(yy.numpy()); row_indices.append(rr.numpy())
                    if step%50==0: print(f'  {panel}: {step}/{len(dl)} batches')
        finally:
            handle.remove()
        X=np.concatenate(features).astype('float32')
        y=np.concatenate(labels).astype('int64')
        r=np.concatenate(row_indices).astype('int64')
        if not np.isfinite(X).all():raise ValueError(f'{panel}: NaN/Inf in features, not saved')
        np.savez_compressed(output,features=X,labels=y,row_indices=r)
        meta={'dataset':panel, 'n':int(len(y)), 'positive':int((y==1).sum()),
              'negative':int((y==0).sum()),'feature_width':int(X.shape[1]),
              'files':[str(p) for p in source_files[panel]],'checkpoint':str(ckpt),
              'chosen_fold':args.yeast_fold if panel=='Yeast' else args.ms_fold,
              'seed':args.seed,'original_kfold_seed':42,'kfold_splits':5,
              'max_length':800,'selection':'exact pair not present in checkpoint training fold; then sample per class',
              'comparison':'All five Multi-species panels share the same model trained on Multi-species_40; this is not five separately trained models.',
              'audit':stats[panel]}
        output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        summary[panel]=meta
        print('[SAVED]',output,'shape=',X.shape)
    (out/'run_summary.json').write_text(json.dumps(summary,indent=2))
    print('Embedding extraction finished. Run project_tsne.py with this feature directory.')

def main():
    p=argparse.ArgumentParser(description='TopPPI: extract features.')
    p.add_argument('mode',choices=['audit','extract'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--yeast-files',nargs='+')
    p.add_argument('--ms40-files',nargs='+')
    p.add_argument('--ms40-source',choices=['train_test','merged'],default='train_test',
                   help='Default uses train_40 followed by test_40, as in the manuscript training script.')
    p.add_argument('--ms01-files',nargs='+')
    p.add_argument('--ms10-files',nargs='+')
    p.add_argument('--ms25-files',nargs='+')
    p.add_argument('--msany-files',nargs='+')
    p.add_argument('--yeast-fold',type=int,choices=range(1,6),default=3)
    p.add_argument('--ms-fold',type=int,choices=range(1,6),default=1)
    p.add_argument('--esm-path',default=__import__('os').environ.get('TOPPPI_ESM_PATH',str(Path(__file__).resolve().parents[2]/'pretrained/esm2_t33_650M_UR50D')))
    p.add_argument('--output-dir',default=str(Path(__file__).resolve().parents[2]/'outputs/visualization/features'))
    p.add_argument('--yeast-ckpt', required=True)
    p.add_argument('--ms-ckpt', required=True)
    p.add_argument('--per-class',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.batch_size<1:p.error('--batch-size must be >=1')
    prepared=prepare(args)
    if args.mode=='extract':extract_features(args,prepared)

if __name__=='__main__':main()
