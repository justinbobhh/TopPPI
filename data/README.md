# Benchmark inputs

All datasets listed below are included as complete files, with original row order and bytes preserved. Saved Yeast/Multi-species fold files and the twelve PDB structures used by Figure 6 are also included. `input_checksums.json` records their source paths, sizes and SHA-256 checksums.

```text
data/
  process_yeast/data_all.csv
  process_yeast/train.csv, test.csv
  process_yeast/5_fold/train1.csv ... train5.csv
  process_yeast/5_fold/valid1.csv ... valid5.csv
  process_multispecies/train_01.csv, test_01.csv
  process_multispecies/train_10.csv, test_10.csv
  process_multispecies/train_25.csv, test_25.csv
  process_multispecies/train_40.csv, test_40.csv
  process_multispecies/train_any.csv, test_any.csv
  process_multispecies/dataset40_merged.csv
  process_multispecies/kfold_5_splits/fold_1_train.csv ... fold_5_val.csv
  process_multiclass/train.csv, test.csv
  process_multiclass/dataset_merged_all.csv
  human_virus/VF1/VirusAppear_VF1_Train.csv
  human_virus/VF1/VirusAppear_VF1_Test.csv
  human_virus/VF1/VirusAppear_VF1_C1Virus.csv
  human_virus/VF1/VirusAppear_VF1_C2Virus.csv
  ... VF2 through VF5
  pdb/                         Twelve complete PDB files, matching complex_manifest.csv
```

Original local sources: `data/SSPPI/data/process_*` and `DeepGNHV-master/data/my_csv_data/my_csv_data`. `prepare_data.py` copies the required inputs to this layout without modifying originals. The nested human-virus folder contains the merged positive/negative splits.

CSV header: `protein_a_id,protein_b_id,seq_a,seq_b,label`. Human-virus CSVs require `seq_a,seq_b,label`. Preserve exact row order to reconstruct the paper's KFold partitions.

Table 1: Yeast = 11,188 records / 2,497 proteins; Multi-species = 47,928 / 9,739; Multi-class = 68,729 / 4,283. These counts were checked against the local processed CSVs.

Original pair annotations and offline sequence dictionaries are now included under [`raw/`](raw/README.md). Use the root `data_process.py` to regenerate the primary benchmark splits without online sequence retrieval. The Multi-class dictionary is derived from the sequences retained in the original processed snapshot. Historical preprocessing code under `preprocess/` retains the original path/network-based conversion for reference. Already processed benchmark files can be used directly.

The primary Multi-species training/evaluation code reads `train_40.csv` then `test_40.csv` and constructs its own KFold partitions. The archival `dataset40_merged.csv` has a different row order and is not used by those entries. Newly converted `dataset40_train_test_merged.csv` explicitly follows train-then-test order. Original CSV snapshots are unchanged.

Benchmark CSVs are tracked using Git LFS through the repository's `.gitattributes`; they are not excluded by `.gitignore`. The local ZIP contains the complete CSV/PDB files, not LFS pointers. Use Git LFS when uploading this data-inclusive repository to GitHub. Small figure summary tables and case-study inputs are kept beside their corresponding experiment code.
