# Original pair annotations and sequence snapshots

| Folder | Pair annotations | Sequence dictionary |
| --- | --- | --- |
| `yeast/` | Original train/test; original source five-fold annotations in `source_5_fold/` | Original `protein.dictionary.tsv`, stored as `sequences.tsv` |
| `multispecies/` | Original train/test for 01, 10, 25, 40 and any identity subsets | Original `CeleganDrosophilaEcoli.dictionary.tsv`, stored as `sequences.tsv` |
| `multiclass/` | Original train/test interaction-type annotations | Derived snapshot of the sequences retained in the manuscript's processed train/test CSVs |

The original pair annotations were reused from [SSPPI](https://github.com/bixiangpeng/SSPPI/). Yeast and Multi-species sequence dictionaries are the original local benchmark dictionaries. `manifest.json` records each file's local source, size and SHA-256. The Multi-class dictionary is explicitly a derived retained-sequence snapshot, not a new download of all original proteins.

Run the offline converter from the repository root:

```bash
python data_process.py --task yeast
python data_process.py --task multispecies
python data_process.py --task multiclass
```

Results go to `outputs/preprocessed/TASK/`. Missing sequences are counted in `conversion_report.json`. The Multi-class retained sequence snapshot excludes the same unavailable records as the paper data. Each generated split has been compared with its bundled processed CSV, including row order.

The Multi-species converter additionally writes `dataset40_train_test_merged.csv`, concatenating train then test exactly as the training code does. The older bundled `process_multispecies/dataset40_merged.csv` is an archival snapshot with a different row order and is not the input used by the primary training/inference entry points. Source-provided Yeast folds are retained as original annotations; primary experiments reconstruct their own KFold partitions with seed 42.

For custom datasets, supply `--dictionary`, `--pairs` and `--output-dir`. Pair files contain three whitespace-separated columns: protein A ID, protein B ID and integer label. Dictionaries contain two tab-separated columns: protein ID and amino acid sequence. The converter preserves the supplied pair order.
