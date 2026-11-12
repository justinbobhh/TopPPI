import argparse
import json
import os
from pathlib import Path
from config import ROOT, config

def prepare_training(task):
    parser = argparse.ArgumentParser(description=f'TopPPI {task}: manuscript training protocol.')
    parser.add_argument('--esm-path', type=Path, default=Path(config.model_name))
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / task)
    parser.add_argument('--show-config', '--dry-run', action='store_true', dest='show_config',
                        help='Print the effective paper configuration without loading models or training.')
    args = parser.parse_args()
    config.select_task(task)
    config.model_name = str(args.esm_path.expanduser().resolve())
    config.data_root = str(args.data_root.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    if int(os.environ.get('RANK', 0)) == 0:
        print(json.dumps({'task': task, 'configuration': vars(config), 'output_dir': str(output)}, indent=2), flush=True)
    if args.show_config:
        raise SystemExit(0)
    if not Path(config.model_name).is_dir():
        parser.error('ESM-2 directory not found. Supply --esm-path or TOPPPI_ESM_PATH.')
    required = {
        'yeast': ['process_yeast/data_all.csv'],
        'multispecies': ['process_multispecies/train_40.csv', 'process_multispecies/test_40.csv'],
        'multiclass': ['process_multiclass/train.csv', 'process_multiclass/test.csv'],
        'human_virus': [f'human_virus/VF{n}/VirusAppear_VF{n}_{split}.csv'
                        for n in range(1, 6) for split in ('Train', 'C2Virus')],
    }[task]
    missing = [p for p in required if not (Path(config.data_root) / p).is_file()]
    if missing:
        parser.error('Missing data: ' + ', '.join(missing))
    output.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get('RANK', 0)) == 0:
        (output / 'run_config.json').write_text(json.dumps(vars(config), indent=2), encoding='utf8')
    os.chdir(output)
