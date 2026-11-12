import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc as trapezoidal_auc
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

plt.rcParams.update({'font.family': 'sans-serif', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})
COLORS = ['#27658c', '#d48739', '#4c8a74', '#956dae', '#b45459']

def load_fold(root, fold, split):
    stem = f'{fold}_{split}_full'
    metrics_path = root / 'metrics' / f'{stem}.json'
    pred_path = root / 'predictions' / f'{stem}.csv'
    meta = json.loads(metrics_path.read_text(encoding='utf-8'))
    if (meta.get('is_partial') or meta.get('fold') != fold or
            meta.get('split') != split or meta.get('run_tag') != 'full'):
        raise ValueError(f'Incomplete or mismatched evaluation: {metrics_path}')
    labels, scores = [], []
    with pred_path.open(encoding='utf-8', newline='') as f:
        for index, row in enumerate(csv.DictReader(f)):
            if int(row['row_index']) != index:
                raise ValueError(f'Missing or out-of-order row in {pred_path}')
            labels.append(int(row['label']))
            scores.append(float(row['probability']))
    y, p = np.asarray(labels), np.asarray(scores)
    if (len(y) != meta['source_row_count'] or len(y) != meta['evaluated_row_count']
            or set(y.tolist()) != {0, 1} or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError(f'Incomplete or invalid predictions: {pred_path}')
    auc, ap = roc_auc_score(y, p), average_precision_score(y, p)
    if not np.isclose(auc, meta['metrics']['AUROC'], atol=1e-7) or not np.isclose(ap, meta['metrics']['AUPRC'], atol=1e-7):
        raise ValueError(f'Predictions do not match stored metrics: {pred_path}')
    return y, p, auc, ap

def plot(root, split, folds, output):
    datasets = [(fold, *load_fold(root, fold, split)) for fold in folds]
    output.mkdir(parents=True, exist_ok=True)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(8.0, 3.5), layout='constrained')
    rows, summary = [], []
    for i, (fold, y, p, auc, ap) in enumerate(datasets):
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        color = COLORS[i % len(COLORS)]
        ax_roc.plot(fpr, tpr, color=color, lw=1.25, label=f'{fold}  {auc:.3f}')
        ax_pr.plot(recall, precision, color=color, lw=1.25, label=f'{fold}  {ap:.3f}')
        rows.extend((fold, 'ROC', x, z) for x, z in zip(fpr, tpr))
        rows.extend((fold, 'PR', x, z) for x, z in zip(recall, precision))
        pr_auc = float(trapezoidal_auc(recall, precision))
        summary.append((fold, len(y), y.mean(), auc, ap, pr_auc))
    ax_roc.plot([0, 1], [0, 1], '--', color='0.65', lw=0.8)
    prevalence = [y.mean() for _, y, _, _, _ in datasets]
    if np.allclose(prevalence, prevalence[0]):
        ax_pr.axhline(prevalence[0], ls='--', color='0.65', lw=0.8, label=f'Prevalence  {prevalence[0]:.3f}')
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False-positive rate', ylabel='True-positive rate', title='ROC')
    ax_pr.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall', ylabel='Precision', title='Precision–recall')
    ax_roc.legend(title='Fold  AUROC', loc='lower right', fontsize=8)
    ax_pr.legend(title='Fold  average precision', loc='lower left', fontsize=8)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{split}_five_fold_curves.{suffix}', dpi=300)
    plt.close(fig)
    with (output / f'{split}_curve_points.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['fold', 'curve', 'x', 'y']); w.writerows(rows)
    with (output / f'{split}_curve_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['fold', 'n', 'positive_prevalence', 'AUROC', 'average_precision', 'PR_AUC_trapezoidal']); w.writerows(summary)
    print(f'Figures and curve data saved to: {output}')
    if split == 'C2Virus':
        print('C2Virus was used for historical checkpoint selection; curves are selection-set estimates.')

def main():
    parser = argparse.ArgumentParser(description='TopPPI: plot saved curves.')
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'evaluation_outputs')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--split', choices=['C1Virus', 'C2Virus'], default='C2Virus')
    parser.add_argument('--folds', nargs='+', default=['VF1', 'VF2', 'VF3', 'VF4', 'VF5'])
    args = parser.parse_args()
    plot(args.results, args.split, args.folds, args.output or args.results / 'figures')

if __name__ == '__main__':
    main()
