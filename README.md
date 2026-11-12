# TopPPI

**Topology-guided partner-conditioned learning for sequence-only protein–protein interaction prediction**

TopPPI adapts ESM-2 with LoRA, refines sequence-derived token graphs with graph attention, and introduces partner information through bidirectional cross-protein attention before pooling. Separate models support binary interaction prediction and seven-class interaction typing.

[TopPPI architecture (PDF)](assets/TopPPI_Figure1.pdf)

## Contents

- [Repository contents](#repository-contents)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Pretrained models](#pretrained-models)
- [Inference and evaluation](#inference-and-evaluation)
- [Ablations, figures and case studies](#ablations-figures-and-case-studies)
- [Reproducibility](#reproducibility)

## Repository contents

```text
TopPPI/
  train.py                    One training entry point for four tasks
  inference.py                Custom pair prediction and benchmark evaluation
  data_process.py             Offline raw-annotation to sequence-pair conversion
  check_assets.py             Verify data and model files
  core/                       Model, graph/attention layers, losses and weight loader
  configs/                    Four manuscript task configurations
  data/
    raw/                      Original pair annotations and sequence dictionaries
    process_yeast/            Processed Yeast data and saved fold files
    process_multispecies/     Five identity subsets and saved fold files
    process_multiclass/       Processed seven-class interaction data
    human_virus/              VF1–VF5: Train, Test, C1Virus and C2Virus
    pdb/                      Twelve structures for the attention case study
  checkpoints/                Four selected task models and their provenance
  pretrained/                 Offline ESM-2 backbone and tokenizer
  examples/                   Three real example sequence pairs
  experiment/                 Original paper-specific experiment and plot workflows
  results/                    Saved result catalogue and manuscript figure PDFs
  assets/                     Architecture figure PDF
```

Experiment-specific training, evaluation and plotting workflows are under `experiment/`.

## Installation

The recorded training/evaluation stack is Python **3.10.19**, PyTorch **2.1.2+cu121**, PyG **2.4.0**, Transformers **4.36.0**, PEFT **0.7.0**, NumPy **1.26.4** and scikit-learn **1.3.0**. Distributed training uses Linux, NVIDIA GPUs and NCCL.

```bash
conda create -n topppi python=3.10.19
conda activate topppi
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The CUDA installation follows the [official PyTorch 2.1.2 instructions](https://pytorch.org/get-started/previous-versions/#v212). The 15 pinned packages include the recorded core versions, seven local analysis/plotting versions, and compatible fixed choices for Accelerate and Requests. This is not a complete historical environment lock.

CSV datasets, fold models and backbone shards use Git LFS. After cloning the repository, fetch the actual files and check the package:

```bash
git lfs install
git lfs pull
python check_assets.py
python check_assets.py --full
```

The delivered local ZIP already contains the actual files. `--full` verifies SHA-256 for 107 data/model assets. The offline backbone is selected automatically; an alternative can be supplied through `--esm-path` or `TOPPPI_ESM_PATH`.

PyMOL is additionally needed to render the structure panels.

## Data preparation

The package includes the processed benchmark CSVs, saved splits, 12 PDB structures, original pair annotations and offline sequence dictionaries. Input CSV columns are `protein_a_id,protein_b_id,seq_a,seq_b,label`. Binary labels are 0/1; interaction-type labels are 0–6.

| Benchmark | Protein IDs | Pair–label records | Main source used in training |
| --- | ---: | ---: | --- |
| Yeast | 2,497 | 11,188 | `data/process_yeast/data_all.csv` |
| Multi-species <40% | 9,739 | 47,928 | `train_40.csv`, then `test_40.csv` |
| Multi-class | 4,283 | 68,729 | `train.csv`, then `test.csv` |
| Human–virus | Per supplied VF split | VF1–VF5 supplied files | `data/human_virus/` |

The included processed data can be used directly. To rebuild the three primary benchmarks offline:

```bash
python data_process.py --task yeast
python data_process.py --task multispecies
python data_process.py --task multiclass
```

Generated files and conversion reports go to `outputs/preprocessed/TASK/`. All generated primary split records and their order were checked against the included snapshots. The Multi-class dictionary contains the sequences retained by the original preprocessing, so no new Ensembl download is required. For custom annotation files, use `--dictionary`, `--pairs` and `--output-dir`.

See [processed-data details](data/README.md) and [raw-data provenance](data/raw/README.md). The older Multi-species merged snapshot has a different order; the training/evaluation entries use the documented train-then-test order. Original source files are preserved.

## Training

Each task selects the paper configuration and the matching data automatically:

```bash
python train.py --task yeast
python train.py --task multispecies
python train.py --task multiclass
python train.py --task human_virus
```

The default is four GPU processes, with batch size eight per process. Use `--nproc-per-node 1` for one GPU. Inspect the settings without training:

```bash
python train.py --task multiclass --show-config
```

| Task | Maximum epochs | Learning rate | GAT width | Checkpoint selection |
| --- | ---: | ---: | ---: | --- |
| Yeast | 25 | 1e-4 | 512 | Validation F1 |
| Multi-species | 12 | 5e-5 | 512 | Validation AUROC |
| Multi-class | 20 | 1e-4 | 128 | Validation macro-F1 |
| Human–virus | 10 | 1e-4 | 128 | C2Virus AP |

All tasks use the 800-token limit, including special tokens. The primary tasks use shuffled record-level five-fold KFold with seed 42. Training outputs are saved under `outputs/TASK/`; bundled publication checkpoints remain in `checkpoints/`. Optional path arguments are `--esm-path`, `--data-root` and `--output-dir`.

## Pretrained models

Four author-selected task checkpoints are included, together with one shared offline ESM-2 backbone. Fold choices are Yeast 3, Multi-species 1, Multi-class 3 and Human–virus VF1:

| Task | Included trained folds | Approximate size per fold |
| --- | --- | ---: |
| Yeast | [Fold 3](checkpoints/yeast/best_model_fold_3.pth) | 74.2 MiB |
| Multi-species <40% | [Fold 1](checkpoints/multispecies/multi_best_model_fold_1.pth) | 74.2 MiB |
| Multi-class | [Fold 3](checkpoints/multiclass/best_model_mc_fold_3.pth) | 24.4 MiB |
| Human–virus | [VF1](checkpoints/human_virus/best_model_VF1.pt) | 19.3 MiB |

The common backbone is stored once in six shards under [`pretrained/`](pretrained/README.md). Each fold retains all LoRA/task parameters and any backbone tensor differing from that common copy. Every reconstructed tensor was compared with its original full checkpoint and matched exactly; no precision conversion or retraining was performed. [`checkpoints/manifest.json`](checkpoints/manifest.json) links each exported file to the original checkpoint checksum.

`core/checkpoints.py` reconstructs the complete state dictionary and all model loaders keep strict loading. Original full checkpoints are also supported. For a model-only integrity check:

```bash
python inference.py --task yeast --check-only
```

See [checkpoint details](checkpoints/README.md) for the format and the separate historical Figure 5 limitation.

## Inference and evaluation

Predict the supplied example sequence pairs:

```bash
python inference.py --task yeast --input examples/protein_pairs.csv --device cuda
```

Custom inputs need `seq_a,seq_b`; IDs and labels are optional. If labels are provided, every row must have a valid label. Results include per-pair predictions and a configuration/metrics JSON under `outputs/inference/TASK/`.

Evaluate each supplied model on its corresponding validation fold. The fold is selected automatically:

```bash
python inference.py --task yeast --amp float16
python inference.py --task multispecies --amp float16
python inference.py --task multiclass --amp float16
python inference.py --task human_virus --split C2Virus
```

The primary benchmark commands reconstruct the exact record partitions from the original concatenation order. Multi-class reports support-weighted metrics and macro-F1/macro-AUROC. The paper’s five-fold summaries require all five original weights or retraining; the four-model release supports evaluation of the supplied folds. Training still generates all five folds. When all five are supplied, summaries use population standard deviation. These are selected-validation estimates; C2Virus was used for human–virus checkpoint selection. `--limit` runs only a subset and is intended for a smoke check.

For the original detailed evaluators, use `experiment/main/evaluate_multiclass.py` and `experiment/human_virus/evaluate.py`; they also accept the bundled shared-backbone checkpoints. The latter can evaluate both C1Virus and C2Virus and export curve inputs.

## Ablations, figures and case studies

Saved summaries, coordinates, patch scores and network predictions are included beside each experiment. To redraw the figures without model inference:

```bash
python experiment/human_virus/plot_curves.py
python experiment/sensitivity/plot_sensitivity.py
python experiment/visualization/plot_tsne.py
python experiment/case_study/wnt/plot_network.py --model-npz experiment/case_study/wnt/results/predictions.npz --threshold 0.5 --model-only --no-annotation
```

Ablation example, preserving the paper's Yeast fold 4 / 20-epoch protocol:

```bash
torchrun --standalone --nproc_per_node=4 experiment/ablation/run_ablation.py --variant full --only-fold 4 --epochs 20
```

Other variants are `attention_only`, `knn_only`, `wo_lora`, `wo_cross_attn` and `wo_struct_gat`. The [results catalogue](results/README.md) links to existing outputs and manuscript PDFs. Use each experiment script’s `--help` to inspect feature-extraction, sensitivity and structural-rendering arguments. Figures 6–7 use the included Multi-species fold 1 model; PDB coordinates are used for the post-prediction case analysis.

## Reproducibility

Completed checks include full SHA-256 verification of 107 data/model assets, lossless reconstruction of the supplied checkpoints, offline regeneration of the three primary data collections and command/configuration checks. Offline forward smoke tests passed for ESM-2 and the complete binary/seven-class TopPPI models, using short prefixes of real example sequences. These checks used an existing secondary CPU environment, which differs from the recorded GPU software stack. Full-length TopPPI GPU benchmark training/inference has not been rerun during packaging.

Figure 5's historical feature-export metadata did not record checkpoint hashes, and same-named local backups differ. Its saved paper coordinates are included for exact redrawing; do not substitute an arbitrary fold weight and claim to reproduce that original extraction. Figures 6–7 have verified source-checkpoint identity.

## Acknowledgements


Benchmark annotations and the Wnt case inputs were reused from [SSPPI](https://github.com/bixiangpeng/SSPPI/). The pretrained encoder originates from [ESM-2](https://huggingface.co/facebook/esm2_t33_650M_UR50D); its original model card and upstream license are retained in `pretrained/`. Human–virus adaptations retain their [third-party notice](experiment/human_virus/THIRD_PARTY_NOTICE.txt). Publication and licensing details are not included in this review release.
