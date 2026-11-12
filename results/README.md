# Published experiment outputs

Saved numeric outputs are kept beside the experiment that produced them so each plotting script can find its own inputs.

| Paper item | Saved inputs/results | Reproduction entry |
| --- | --- | --- |
| Human-virus ROC/PR curves (Figure 2) | [`experiment/human_virus/results/curves/`](../experiment/human_virus/results/curves/) | `python experiment/human_virus/plot_curves.py` |
| Sensitivity (Figures 3–4) | [`experiment/sensitivity/results/`](../experiment/sensitivity/results/) | `python experiment/sensitivity/plot_sensitivity.py` |
| Pair representations (Figure 5) | [`experiment/visualization/results/`](../experiment/visualization/results/) | `python experiment/visualization/plot_tsne.py` |
| Structure attention (Figure 6) | [`experiment/case_study/structures/results/`](../experiment/case_study/structures/results/) | `prepare_pymol.py`, PyMOL, then `assemble_panel.py` in that experiment folder |
| Wnt predictions (Figure 7) | [`experiment/case_study/wnt/results/`](../experiment/case_study/wnt/results/) | `plot_network.py` in that experiment folder |
| Component ablation (Table 7) | [`experiment/ablation/results/`](../experiment/ablation/results/) | `run_ablation.py` in that experiment folder |

`figures/` contains the six existing experiment PDFs used in the manuscript. The current architecture is in [`assets/TopPPI_Figure1.pdf`](../assets/TopPPI_Figure1.pdf). `figures/manifest.json` records the original local PDF paths and checksums.

The four supplied task checkpoints support new evaluations through `inference.py` and the original experiment-specific evaluators. Saved plots/coordinates are historical outputs. Figure 5's original extraction checkpoint cannot be uniquely identified from the old metadata; redraw its saved coordinates to reproduce the existing visualization. Figures 6–7 have verified original checkpoint provenance. No new model training or benchmark values are asserted by packaging these outputs.
