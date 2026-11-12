# Shared ESM-2 backbone

`esm2_t33_650M_UR50D/` contains the offline encoder and tokenizer used by the local experiments, originating from [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D).

The original PyTorch weight file is losslessly split into six shards, each below 500 MiB. `pytorch_model.bin.index.json` is the standard Transformers shard index. The configuration, vocabulary and tokenizer files are included. `manifest.json` records the original file checksum and each shard checksum. Every shard tensor was compared with the original local backbone.

This directory can be passed directly to Transformers `from_pretrained(..., local_files_only=True)`. The project defaults to it, so downloading the model again is unnecessary after retrieving all Git LFS objects. The model's original card is retained in its subfolder; it identifies the upstream model's MIT license, which does not assign a license to the TopPPI project.
