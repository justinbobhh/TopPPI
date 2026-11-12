import csv
import argparse
import hashlib
import json
from pathlib import Path
from common import load_chain, read_complex_manifest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'results'
OUT = HERE.parents[2] / 'outputs/case_study/structures'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    global SOURCE, OUT
    parser = argparse.ArgumentParser(description='TopPPI: prepare pymol.')
    parser.add_argument('--results-dir', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--pdb-dir', type=Path, default=HERE.parents[2] / 'data/pdb')
    args = parser.parse_args()
    SOURCE, OUT = args.results_dir.resolve(), args.output_dir.resolve()
    manifest = read_complex_manifest(HERE / 'complex_manifest.csv')
    run = json.loads((SOURCE / 'run_manifest.json').read_text(encoding='utf-8'))
    if run['status'] != 'complete': raise ValueError('Fold 1 run is not complete')
    patches, residues = read(SOURCE/'patch_points.csv'), read(SOURCE/'residue_points.csv')
    checks = {r['pdb_id']:r for r in read(SOURCE/'capture_checks.csv')}
    if set(run['completed']) != {r['pdb_id'] for r in manifest}: raise ValueError('Unexpected PDB panel')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    commands=['reinitialize','bg_color white','set ray_opaque_background, 1','set antialias, 2',
              'set_color chain_blue, [0.10, 0.35, 0.62]',
              'set_color chain_yellow, [0.86, 0.76, 0.42]', 'set_color high_red, [0.72, 0.16, 0.12]']
    selected=[]
    for row in manifest:
        pid=row['pdb_id']; pdb=args.pdb_dir.resolve()/row['pdb_file']
        if hashlib.sha256(pdb.read_bytes()).hexdigest() != checks[pid]['pdb_sha256']:
            raise ValueError(f'{pid}: PDB hash does not match inference')
        clauses=[]
        for role, cid in [('A',row['chain_a']),('B',row['chain_b'])]:
            ps=sorted([p for p in patches if p['pdb_id']==pid and p['chain_role']==role],
                      key=lambda p:(-float(p['attention']),p['patch_id']))[:2]
            if len(ps)!=2: raise ValueError(f'{pid} {role}: expected two patches')
            rr=[r for r in residues if r['pdb_id']==pid and r['chain_role']==role]
            records=load_chain(pdb,cid)
            if [r['residue_key'] for r in rr] != [r.key for r in records]:
                raise ValueError(f'{pid} {cid}: residue alignment mismatch')
            for p in ps:
                matched=[r for r in rr if r['patch_id']==p['patch_id']]
                if len(matched)!=int(p['length']): raise ValueError(f'{pid}: patch length mismatch')
                nums=[]
                for r in matched:
                    _,num,icode=r['residue_key'].split(':')
                    nums.append(num + ('' if icode=='.' else icode))
                clauses.append(f'(chain {cid} and resi {"+".join(nums)})')
                selected.append({'pdb_id':pid,'chain_role':role,'chain_id':cid,'patch_id':p['patch_id'],
                                 'attention':p['attention'],'residue_count':len(matched),'pdb_residues':'+'.join(nums)})
        commands += [f'load "{pdb.as_posix()}", complex',
            f'remove complex and not (chain {row["chain_a"]} or chain {row["chain_b"]})',
            'remove solvent','hide everything','show cartoon, complex',
            f'color chain_blue, complex and chain {row["chain_a"]}',
            f'color chain_yellow, complex and chain {row["chain_b"]}',
            'select high, complex and ('+' or '.join(clauses)+')','color high_red, high',
            'orient complex','zoom complex, 5','python','from pymol import cmd',
            f'cmd.png({(OUT/"figures"/(pid+"_cartoon.png")).as_posix()!r}, width=2400, height=1800, dpi=300, ray=1)',
            'python end','hide cartoon','show surface, complex','python','from pymol import cmd',
            f'cmd.png({(OUT/"figures"/(pid+"_surface.png")).as_posix()!r}, width=2400, height=1800, dpi=300, ray=1)',
            'python end','delete high','delete complex']
    (OUT/'render_fold1.pml').write_text('\n'.join(commands)+'\n',encoding='utf8')
    with (OUT/'fold1_top_patches.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0])); w.writeheader(); w.writerows(selected)
    print(f'Prepared {len(selected)} patch selections in {OUT}')

if __name__=='__main__': main()
