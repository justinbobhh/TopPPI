# Trained models

One author-selected model per task is included. Filenames retain the original fold identity; selection was confirmed by the authors rather than inferred from file numbering:

```text
checkpoints/
  yeast/best_model_fold_3.pth
  multispecies/multi_best_model_fold_1.pth
  multiclass/best_model_mc_fold_3.pth
  human_virus/best_model_VF1.pt
  manifest.json
```

The shared offline encoder/tokenizer is in `../pretrained/esm2_t33_650M_UR50D/`. These are real trained parameters from the original local experiments, repackaged without numerical conversion or retraining.

## Exact shared-backbone format

The original full checkpoints repeat approximately the same 650M-parameter frozen ESM-2 model. The distribution stores that backbone once and retains every LoRA tensor, task tensor and any backbone value differing from the common copy inside each fold file. Nothing is discarded based only on its name or assumed to be frozen: equality with the shared tensor is checked first.

Each checkpoint includes its task/fold, original full-file SHA-256, the exact original state key set, retained tensors and an explicit mapping to the shared tensors. Every reconstructed tensor was verified equal in shape, dtype and value to its original. `manifest.json` records exported-file checksums and source provenance. The original experiment files are unchanged.

Use the strict loader:

```python
from pathlib import Path
import sys
sys.path.insert(0, 'core')
from checkpoints import read_checkpoint_state
state = read_checkpoint_state(Path('checkpoints/yeast/best_model_fold_3.pth'), expected_task='yeast')
model.load_state_dict(state, strict=True)
```

`train.py`, `inference.py` and the original paper-specific evaluation/extraction workflows use the matching settings and accept these files. The loader also supports original full checkpoints. New training runs continue writing the original full checkpoint format under `outputs/`.

## Task correspondence

Yeast and Multi-species use GAT width 512 / classifier input 2048; Multi-class and human–virus use width 128 / classifier input 512. Multi-class output width is seven, while binary outputs have width one. Table 5 evaluates support-weighted metrics using macro-F1-selected multiclass folds. Only fold 3 is bundled here; reproducing the complete five-fold aggregate requires the other original weights or retraining. The default inference route uses the supplied task-specific fold.

Figures 6–7 use the supplied Multi-species fold 1. Its original full-checkpoint SHA-256 is `79c3abc84709a6b4e9c52e626a3006ec3d52f5fd2799af2b2f0b2abacf32980f`, matching the historical run records. The compact file has its own checksum; the lossless reconstruction links the two.

Figure 5's original extraction metadata identifies Yeast fold 3 and Multi-species fold 1 by filename but does not record their hashes. Different local backups share those names. The original extraction identity remains unresolved; the supplied primary-task checkpoints are not relabelled as those historical extraction weights. Use the included coordinates to redraw Figure 5, or pass explicitly identified extraction weights to `extract_features.py` for a new extraction.

The model files and shared shards are tracked by Git LFS. Run `git lfs pull` after cloning and `python check_assets.py --full` to verify complete downloads. The local release ZIP contains the actual files.
