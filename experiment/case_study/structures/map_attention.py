import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import attention_utils as base
from attention_utils import incoming_scores, patch_rows
from common import (atomic_write_csv, atomic_write_json, chain_sequence, file_identity,
                    load_chain, read_complex_manifest, resolve_pdb_path)

def pearson(rows):
    if len(rows) < 3:
        return None
    x=np.array([r['attention'] for r in rows], dtype=float)
    y=np.array([r['distance_A'] for r in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite attention or distance')
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0,1])

def main():
    p=argparse.ArgumentParser(description='TopPPI: map attention.')
    for name in ('model-dir','esm-dir','checkpoint','manifest','pdb-dir','output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--amp', choices=['none','fp16','bf16'], default='fp16')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--capture-atol', type=float, default=0.002)
    args=p.parse_args()

    args.max_length = 800
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Output not empty: {args.output_dir}; choose a new output directory')
    for path in (args.checkpoint, args.model_dir/'model.py', args.model_dir/'layers.py', args.model_dir/'config.py'):
        if not path.is_file(): raise FileNotFoundError(path)
    if not args.esm_dir.is_dir(): raise FileNotFoundError(args.esm_dir)
    rows=read_complex_manifest(args.manifest)
    for row in rows:
        pdb=resolve_pdb_path(row,args.pdb_dir)
        for cid in (row['chain_a'],row['chain_b']):
            n=len(load_chain(pdb,cid))
            if not 26 <= n <= 798: raise ValueError(f'{row["pdb_id"]} {cid}: unsupported length {n}')
    torch, tokenizer_class, _, model_class=base.import_runtime(args)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA environment required')
    device=torch.device(args.device); torch.cuda.set_device(device)
    torch.manual_seed(20261031); torch.cuda.manual_seed_all(20261031)
    tokenizer=tokenizer_class.from_pretrained(str(args.esm_dir.resolve()),local_files_only=True)
    model=model_class(); base.load_checkpoint_strict(model,args.checkpoint,torch); model.eval().to(device)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running','checkpoint':file_identity(args.checkpoint,True),
      'model_source':file_identity(args.model_dir/'model.py',True),
      'layers_source':file_identity(args.model_dir/'layers.py',True),
      'config_source':file_identity(args.model_dir/'config.py',True),
      'manifest':file_identity(args.manifest,True),'script':file_identity(Path(__file__),True),
      'amp':args.amp,'capture_atol':args.capture_atol,'gpu':torch.cuda.get_device_name(device),
      'torch':torch.__version__,'python':platform.python_version(),'esm_dir':str(args.esm_dir.resolve()),
      'max_length':800,'padding':'max_length','patch_size':25,
      'score':'mean incoming cross-attention across heads and valid queries; each valid-key row renormalized',
      'distance':'C-alpha to other chain C-alpha centroid, fallback missing CA to mean residue atom coordinates',
      'both_chains':True,'completed':[]}
    atomic_write_json(args.output_dir/'run_manifest.json',manifest)
    all_res, all_patch, checks=[],[],[]
    for row in rows:
        pid=row['pdb_id']; pdb=resolve_pdb_path(row,args.pdb_dir)
        chains=[load_chain(pdb,row['chain_a']),load_chain(pdb,row['chain_b'])]
        inputs=[]; positions=[]
        for c in chains:
            inp,pos=base.tokenize_chain(tokenizer,chain_sequence(c),800,'max_length',torch)
            inputs.append(base.to_device(inp,device)); positions.append(pos.cpu().numpy())
        captured={}
        def capture(name):
            def hook(module, inp, out):
                if not isinstance(out,tuple) or out[1] is None: raise RuntimeError(f'{name}: no attention')
                captured[name]=out[1].detach().float().cpu().numpy()
            return hook
        with torch.inference_mode(),base.autocast_context(torch,args.amp):
            direct=float(model(*inputs)[0].reshape(-1)[0].float().cpu())
        handles=[model.cross_attn_v2h.mha.register_forward_hook(capture('ab')),
                 model.cross_attn_h2v.mha.register_forward_hook(capture('ba'))]
        try:
            with torch.inference_mode(),base.autocast_context(torch,args.amp):
                observed=float(model(*inputs)[0].reshape(-1)[0].float().cpu())
        finally:
            for h in handles: h.remove()
        difference=abs(direct-observed)
        if not np.isfinite([direct,observed]).all() or difference > args.capture_atol:
            raise RuntimeError(f'{pid}: capture logit difference {difference} > {args.capture_atol}')
        if set(captured)!={'ab','ba'}: raise RuntimeError(f'{pid}: missing attention')
        scores=[incoming_scores(captured['ba'][0],positions[1],positions[0]),
                incoming_scores(captured['ab'][0],positions[0],positions[1])]
        coords=[np.stack([r.ca_coord if r.ca_coord is not None else
                np.mean([a.coord for a in r.atoms],axis=0) for r in c]) for c in chains]
        for i,role in enumerate(('A','B')):
            d=np.linalg.norm(coords[i]-coords[1-i].mean(axis=0),axis=1)
            for j,(res,score,dist) in enumerate(zip(chains[i],scores[i],d)):
                all_res.append({'pdb_id':pid,'chain_role':role,'chain_id':res.chain_id,
                    'residue_key':res.key,'patch_id':f'{role}{j//25+1:04d}',
                    'attention':float(score),'distance_A':float(dist)})
            for patch in patch_rows(pid,role,chains[i],scores[i]):
                start=int(patch['start_1'])-1; end=int(patch['end_1'])
                all_patch.append({'pdb_id':pid,'chain_role':role,'chain_id':patch['chain_id'],
                    'patch_id':patch['patch_id'],'length':end-start,'attention':patch['attention_score'],
                    'distance_A':float(d[start:end].mean())})
        checks.append({'pdb_id':pid,'pdb_sha256':file_identity(pdb,True)['sha256'],
                       'direct_logit':direct,'captured_logit':observed,'capture_difference':difference})
        manifest['completed'].append(pid); atomic_write_json(args.output_dir/'run_manifest.json',manifest)
        print(f'[done] {pid}: capture difference={difference:.7f}',flush=True)
    summary=[]
    for pid in ['ALL']+[r['pdb_id'] for r in rows]:
        for role in ('both','A','B'):
            for level, data in (('patch',all_patch),('residue',all_res)):
                selected=[r for r in data if (pid=='ALL' or r['pdb_id']==pid) and (role=='both' or r['chain_role']==role)]
                summary.append({'pdb_id':pid,'chain_role':role,'level':level,'n_points':len(selected),'pearson_r':pearson(selected)})
    atomic_write_csv(pd.DataFrame(all_patch),args.output_dir/'patch_points.csv')
    atomic_write_csv(pd.DataFrame(all_res),args.output_dir/'residue_points.csv')
    atomic_write_csv(pd.DataFrame(summary),args.output_dir/'pearson_summary.csv')
    atomic_write_csv(pd.DataFrame(checks),args.output_dir/'capture_checks.csv')
    manifest['status']='complete'; atomic_write_json(args.output_dir/'run_manifest.json',manifest)
    for r in summary[:2]: print(r,flush=True)
    print(f'All results saved in {args.output_dir}',flush=True)

if __name__=='__main__': main()
