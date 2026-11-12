import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description='TopPPI: prepare data.')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source, destination = args.source_root.resolve(), args.data_root.resolve()
    jobs = []
    for group in ('process_yeast', 'process_multispecies', 'process_multiclass'):
        folder = source / 'data/SSPPI/data' / group
        if not folder.is_dir():
            parser.error(f'Missing original dataset directory: {folder}')
        for p in sorted(folder.rglob('*.csv')):
            if group == 'process_multispecies' and p.name in {'train_cold.csv', 'test_s1.csv', 'test_s2.csv'}:
                continue
            jobs.append((p, destination / group / p.relative_to(folder)))
    for fold in range(1, 6):
        for split in ('Train', 'Test', 'C1Virus', 'C2Virus'):
            rel = f'VF{fold}/VirusAppear_VF{fold}_{split}.csv'
            jobs.append((source / 'DeepGNHV-master/data/my_csv_data/my_csv_data' / rel,
                         destination / 'human_virus' / rel))
    structures = source / 'pdb_dir'
    expected_pdbs = ('1BRS', '1E6E', '1FQJ', '1JK9', '1LB1', '1LFD',
                     '2B10', '2C35', '2I9B', '5JSB', '5LSL', '8HTI')
    jobs += [(structures / f'{pid}.pdb', destination / 'pdb' / f'{pid}.pdb') for pid in expected_pdbs]
    missing = [str(p) for p, _ in jobs if not p.is_file()]
    if missing:
        parser.error('Missing original inputs: ' + ', '.join(missing))
    if args.dry_run:
        print(json.dumps({'files': len(jobs), 'bytes': sum(p.stat().st_size for p, _ in jobs),
                          'destination': str(destination), 'paths': [str(q.relative_to(destination)) for _, q in jobs]}, indent=2))
        return
    manifest = []
    for p, q in jobs:
        value = digest(p)
        if p == q or (q.exists() and digest(q) != value):
            parser.error(f'Destination conflicts with source: {q}')
        manifest.append({'file': q.relative_to(destination).as_posix(),
                         'source': p.relative_to(source).as_posix(),
                         'bytes': p.stat().st_size, 'sha256': value})
    for p, q in jobs:
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists():
            shutil.copy2(p, q)
    (destination / 'input_checksums.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(f'Prepared {len(jobs)} files in {destination}; original inputs unchanged.')

if __name__ == '__main__':
    main()
