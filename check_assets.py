import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description='TopPPI: check assets.')
    p.add_argument('--full',action='store_true',help='Verify SHA-256 in addition to file presence, size and LFS-pointer detection.')
    args=p.parse_args()
    entries=[]
    original=json.loads((ROOT/'data/input_checksums.json').read_text(encoding='utf8'))
    entries += [(ROOT/'data'/r['file'],r['bytes'],r['sha256']) for r in original]
    raw=json.loads((ROOT/'data/raw/manifest.json').read_text(encoding='utf8'))
    entries += [(ROOT/r['path'],r['bytes'],r['sha256']) for r in raw['files']]
    models=json.loads((ROOT/'checkpoints/manifest.json').read_text(encoding='utf8'))
    entries += [(ROOT/r['path'],r['bytes'],r['sha256']) for r in models['checkpoints']]
    base_dir=ROOT/'pretrained/esm2_t33_650M_UR50D'
    base=json.loads((base_dir/'manifest.json').read_text(encoding='utf8'))
    entries += [(base_dir/r['path'],r['size'],r['sha256']) for r in base['shards']]
    failures=[]
    for path,size,expected in entries:
        label=path.relative_to(ROOT).as_posix()
        if not path.is_file():failures.append(f'{label}: missing');continue
        with path.open('rb') as f:prefix=f.read(128)
        if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
            failures.append(f'{label}: Git LFS pointer only; run git lfs pull');continue
        if path.stat().st_size!=size:failures.append(f'{label}: size mismatch');continue
        if args.full and digest(path)!=expected:failures.append(f'{label}: SHA-256 mismatch')
    print(json.dumps({'checked':len(entries),'full_sha256':args.full,'failures':failures,'passed':not failures},indent=2))
    raise SystemExit(1 if failures else 0)

if __name__=='__main__':main()
