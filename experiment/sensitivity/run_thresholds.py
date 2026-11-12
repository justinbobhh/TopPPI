import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

THRESHOLDS = (0.05, 0.10, 0.15)

def threshold_tag(value):

    return f"{value:g}".replace(".", "p")

def parse_args():
    parser = argparse.ArgumentParser(
        description="TopPPI threshold sensitivity experiments"
    )

    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).resolve().parent / "run_layers.py",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="yeast",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data/process_yeast/data_all.csv"),
    )

    parser.add_argument(
        "--fold-dir",
        type=str,
        default="../data/process_yeast/5_fold",
    )

    parser.add_argument(
        "--split-source",
        type=str,
        choices=["fixed", "kfold", "single"],
        default="kfold",
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--gpus", type=str, default="0,3")

    parser.add_argument(
        "--variant",
        type=str,
        default="full",
    )

    parser.add_argument(
        "--attention-layer",
        type=int,
        default=29,
        help="Fixed zero-based ESM-2 layer index.",
    )

    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--gat-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./yeast_threshold_sensitivity"),
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./yeast_threshold_logs"),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing matching experiment.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting training.",
    )

    return parser.parse_args()

def build_command(args, threshold, output_dir):

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",

        "--nproc_per_node",
        str(args.nproc_per_node),

        str(args.script),

        "--variant",
        args.variant,

        "--dataset-name",
        args.dataset_name,

        "--data-path",
        str(args.data_path),

        "--split-source",
        args.split_source,

        "--epochs",
        str(args.epochs),

        "--k-folds",
        str(args.k_folds),

        "--seed",
        str(args.seed),

        "--num-workers",
        str(args.num_workers),

        "--lr",
        str(args.lr),

        "--attention-layer",
        str(args.attention_layer),

        "--knn-k",
        str(args.knn_k),

        "--contact-threshold",
        str(threshold),

        "--gat-heads",
        str(args.gat_heads),

        "--output-dir",
        str(output_dir.resolve()),
    ]

    if args.split_source == "fixed":
        command.extend([
            "--fold-dir",
            str(args.fold_dir),
        ])

    if args.resume:
        command.append("--resume")

    return command

def combine_summaries(args):

    frames = []
    missing_files = []

    for threshold in THRESHOLDS:

        tag = threshold_tag(threshold)

        summary_path = (
            args.output_dir
            / f"threshold_{tag}"
            / f"attention_layer_{args.attention_layer}"
            / f"{args.dataset_name}_{args.variant}_summary.csv"
        )

        if not summary_path.exists():
            missing_files.append(summary_path)
            continue

        df = pd.read_csv(summary_path)

        df.insert(0, "contact_threshold", threshold)
        df.insert(1, "attention_layer_index", args.attention_layer)
        df.insert(2, "knn_k", args.knn_k)
        df.insert(3, "gat_heads", args.gat_heads)

        frames.append(df)

    if frames:

        combined = pd.concat(
            frames,
            ignore_index=True,
        )

        output_path = (
            args.output_dir
            / "threshold_comparison_summary.csv"
        )

        combined.to_csv(
            output_path,
            index=False,
        )

        print("\nCombined summary saved to:")
        print(output_path.resolve())

    if missing_files:

        print("\nWARNING: Some summary files are missing:")

        for path in missing_files:
            print(path)

        print(
            "Only existing summary files were included. "
            "Check the corresponding training logs."
        )

def main():

    args = parse_args()

    args.script = args.script.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()

    if not args.script.is_file():
        raise FileNotFoundError(
            f"Training script not found: {args.script}"
        )

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    if args.split_source == "kfold" and args.k_folds < 2:
        raise ValueError("--k-folds must be >= 2")

    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node must be >= 1")

    if args.attention_layer < 0:
        raise ValueError("--attention-layer must be >= 0")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("TopPPI Threshold Sensitivity Analysis")
    print("=" * 72)

    print(f"Thresholds       : {THRESHOLDS}")
    print(f"Training script  : {args.script}")
    print(f"Dataset          : {args.data_path}")
    print(f"Variant           : {args.variant}")
    print(f"Attention layer  : {args.attention_layer}")
    print(f"KNN k             : {args.knn_k}")
    print(f"GAT heads         : {args.gat_heads}")
    print(f"Epochs / folds   : {args.epochs} / {args.k_folds}")
    print(f"Visible GPUs     : {args.gpus}")
    print(f"DDP processes    : {args.nproc_per_node}")
    print(f"Output directory : {args.output_dir}")
    print("=" * 72)

    for threshold in THRESHOLDS:

        tag = threshold_tag(threshold)

        output_dir = (
            args.output_dir
            / f"threshold_{tag}"
        )

        log_path = (
            args.log_dir
            / f"threshold_{tag}.log"
        )

        command = build_command(
            args,
            threshold,
            output_dir,
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpus

        print("\n" + "#" * 72)
        print(f"Starting contact_threshold = {threshold}")
        print(f"Output directory: {output_dir}")
        print(f"Log file: {log_path}")
        print("Command:")
        print(" ".join(command))
        print("#" * 72)

        if args.dry_run:
            continue

        if not args.resume:
            done_markers = list(
                output_dir.glob(
                    "attention_layer_*/yeast_*_done.json"
                )
            )

            if done_markers:
                raise RuntimeError(
                    f"Existing completed runs found in {output_dir}. "
                    "Check the old results first. Use a new output "
                    "directory or --resume for a matching experiment."
                )

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1
        )

        with log_path.open("w", encoding="utf-8") as log_file:
            for line in process.stdout:

                sys.stdout.write(line)
                sys.stdout.flush()

                log_file.write(line)
                log_file.flush()

        process.wait()

        result = subprocess.CompletedProcess(command, process.returncode)

        if result.returncode != 0:

            print(
                f"\nERROR: threshold {threshold} failed. "
                f"Exit code: {result.returncode}"
            )

            print(f"Check log: {log_path}")

            raise SystemExit(result.returncode)

        print(
            f"Finished threshold = {threshold}"
        )

    if not args.dry_run:

        combine_summaries(args)

        print("\nAll threshold experiments completed.")

if __name__ == "__main__":
    main()
