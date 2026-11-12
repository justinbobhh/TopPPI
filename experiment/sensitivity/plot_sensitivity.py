import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
METRICS = [('Accuracy', 'ACC'), ('Precision', 'Precision'), ('Recall', 'Recall'),
           ('Specificity', 'Specificity'), ('AUC', 'AUC'), ('MCC', 'MCC'), ('F1', 'F1')]
COLORS = ['#59619E', '#AD3829', '#D99345', '#5E9872', '#8B6BA3']

def main():
    parser = argparse.ArgumentParser(description='TopPPI: plot sensitivity.')
    parser.add_argument('--study', choices=['layer', 'threshold', 'both'], default='both')
    parser.add_argument('--output-dir', type=Path, default=HERE.parents[1] / 'outputs/sensitivity')
    parser.add_argument('--results-root', type=Path, help='Optional new-run root containing layers/ and thresholds/ summaries.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    studies = ('layer', 'threshold') if args.study == 'both' else (args.study,)
    for study in studies:
        path = HERE / 'results' / ('layers.csv' if study == 'layer' else 'thresholds.csv')
        if args.results_root:
            folder, filename, column = ('layers', 'attention_layer_comparison_summary.csv', 'attention_layer_index') if study == 'layer' else ('thresholds', 'threshold_comparison_summary.csv', 'contact_threshold')
            path = args.results_root / folder / filename
            data = pd.read_csv(path).rename(columns={column: 'group'})
        else:
            data = pd.read_csv(path)
        values = [25, 27, 29, 31, 32] if study == 'layer' else [.05, .10, .15]
        if set(data['group'].astype(float)) != set(values):
            raise ValueError(f'{path}: parameter labels differ from the paper')
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for ax, (metric, label) in zip(axes.flat, METRICS):
            means, errors = [], []
            for value in values:
                row = data[(data['metric'] == metric) & np.isclose(data['group'].astype(float), value)]
                if len(row) != 1:
                    raise ValueError(f'Expected one real summary for {metric}, {value}')
                means.append(float(row.iloc[0]['mean']) / 100)
                errors.append(float(row.iloc[0]['std']) / 100)
            ax.bar(range(len(values)), means, yerr=errors, capsize=3, color=COLORS[:len(values)], width=.65)
            lower, upper = np.min(np.array(means)-errors), np.max(np.array(means)+errors)
            margin = max((upper-lower)*.2, .0005)
            ax.set_ylim(lower-margin, min(1.005, upper+margin))
            ax.set_title(label)
            ax.set_xticks(range(len(values)), [str(x) for x in values])
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        axes.flat[7].axis('off')
        axes.flat[7].text(.05, .8, 'Yeast: five folds\nMean ± population SD\nTruncated, metric-specific y axes\nValues read from saved summaries', va='top')
        fig.tight_layout()
        stem = args.output_dir / ('figure3_attention_layers' if study == 'layer' else 'figure4_attention_thresholds')
        fig.savefig(stem.with_suffix('.pdf'))
        fig.savefig(stem.with_suffix('.png'), dpi=300)
        plt.close(fig)
        print(stem)

if __name__ == '__main__':
    main()
