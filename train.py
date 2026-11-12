import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='TopPPI: train.')
    parser.add_argument('--task', choices=('yeast', 'multispecies', 'multiclass', 'human_virus'), required=True)
    parser.add_argument('--nproc-per-node', type=int, default=4, help='GPU processes (default 4). Batch size is 8 per process.')
    parser.add_argument('--esm-path', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--show-config', action='store_true', help='Inspect the paper settings without importing PyTorch or starting training.')
    args = parser.parse_args()
    if args.nproc_per_node < 1:
        parser.error('--nproc-per-node must be positive')
    entry = ROOT / ('experiment/human_virus/train.py' if args.task == 'human_virus' else f'experiment/main/train_{args.task}.py')
    command = [sys.executable]
    if not args.show_config:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={args.nproc_per_node}']
    command.append(str(entry))
    for option in ('esm_path', 'data_root', 'output_dir'):
        value = getattr(args, option)
        if value is not None:
            command += ['--' + option.replace('_', '-'), str(value.expanduser().resolve())]
    if args.show_config:
        command.append('--show-config')
    raise SystemExit(subprocess.call(command))

if __name__ == '__main__':
    main()
