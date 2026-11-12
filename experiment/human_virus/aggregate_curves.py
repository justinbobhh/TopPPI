import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from plot_saved_curves import load_fold

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parents[1] / 'outputs/human_virus/evaluation'
REF = ROOT / 'figure_reference'
OUTPUT = ROOT.parents[1] / 'outputs/human_virus/curves'
FOLDS = [f'VF{i}' for i in range(1, 6)]
METHODS = ['DeepGNHV', 'HBFormer', 'TransPPI', 'HVPPI', 'DeepViral', 'CAPHV']
COLORS = ['#D95459', '#E59859', '#D6BE28', '#68A99A', '#77A1C8', '#8773A6']
PANELS = [('A', 'C1Virus', 'ROC'), ('B', 'C1Virus', 'PR'),
          ('C', 'C2Virus', 'ROC'), ('D', 'C2Virus', 'PR')]

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    baseline, reported, removed = {}, {}, []
    with (REF / 'baseline_curves_digitized.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            x, y = float(row['x']), float(row['y'])

            if row['panel'] == 'A' and row['method'] == 'TransPPI' and .255 < x < .280 and y > .85:
                removed.append({'x': x, 'y': y})
                continue
            baseline.setdefault((row['panel'], row['method']), []).append((x, y))
    if len(removed) != 3:
        raise ValueError(f'Unexpected number of extraction errors: {len(removed)}')
    with (REF / 'published_figure2_metrics.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            reported[(row['panel'], row['method'])] = float(row['reported_area'])

    grid = np.linspace(0, 1, 1001)
    means, summaries, fold_rows, raw_curves = {}, {}, [], {}
    for split in ['C1Virus', 'C2Virus']:
        roc_ys, pr_ys, rocs, prs, aps, n = [], [], [], [], [], []
        for fold in FOLDS:
            y, p, _, _ = load_fold(RESULTS, fold, split)
            fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
            precision, recall, _ = precision_recall_curve(y, p)

            roc_ys.append(np.interp(grid, fpr, tpr))
            pr_ys.append(np.interp(grid, recall[::-1], precision[::-1]))
            ra = float(roc_auc_score(y, p)); pa = float(auc(recall, precision))
            ap = float(average_precision_score(y, p))
            rocs.append(ra); prs.append(pa); aps.append(ap); n.append(len(y))
            fold_rows.append([fold, split, len(y), ra, pa, ap])
            raw_curves[(fold, split)] = {'prediction_sha256': hashlib.sha256(
                (RESULTS/'predictions'/f'{fold}_{split}_full.csv').read_bytes()).hexdigest()}
        means[(split, 'ROC')] = np.mean(roc_ys, axis=0)
        means[(split, 'PR')] = np.mean(pr_ys, axis=0)
        summaries[split] = {
            'fold_n': n, 'mean_fold_AUROC': float(np.mean(rocs)),
            'mean_fold_PR_AUC_trapezoidal': float(np.mean(prs)),
            'mean_fold_AP': float(np.mean(aps)),
            'area_of_plotted_mean_ROC': float(auc(grid, means[(split, 'ROC')])),
            'area_of_plotted_mean_PR': float(auc(grid, means[(split, 'PR')]))}

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 9, 'axes.labelweight': 'normal',
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'path.simplify': False, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.3))
    fig.subplots_adjust(left=.105, right=.98, bottom=.08, top=.96, wspace=.28, hspace=.30)
    for ax, (letter, split, curve) in zip(axes.flat, PANELS):
        for method, color in zip(METHODS, COLORS):
            xy = np.array(baseline[(letter, method)])
            if not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
                raise ValueError(f'Invalid reference: {letter} {method}')
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.2, alpha=.95,
                    solid_capstyle='round', solid_joinstyle='round',
                    label=f'{method} ({reported[(letter, method)]:.3f})')
        if curve == 'ROC':
            number = summaries[split]['mean_fold_AUROC']

            xx, yy = np.r_[0, grid], np.r_[0, means[(split, curve)]]
        else:
            number = summaries[split]['mean_fold_PR_AUC_trapezoidal']
            xx, yy = grid, means[(split, curve)]
        ax.plot(xx, yy, color='#3C787B', lw=1.5, alpha=.95, zorder=3,
                solid_capstyle='round', solid_joinstyle='round',
                label=f'TopPPI (mean {number:.3f})')
        ax.set(xlim=(0, 1), ylim=(0, 1),
               xlabel='False positive rate' if curve == 'ROC' else 'Recall',
               ylabel='True positive rate' if curve == 'ROC' else 'Precision')
        ax.set_box_aspect(1/1.06)
        ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6))
        ax.grid(color='#D9D9D9', linewidth=.5, alpha=.7); ax.set_axisbelow(True)
        ax.tick_params(width=.7, length=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.8)
        ax.text(-.15, 1.03, letter, transform=ax.transAxes, fontsize=13,
                fontweight='normal', va='bottom')
        handles, labels = ax.get_legend_handles_labels(); order = [6,0,1,2,3,4,5]
        location = 'lower right' if curve == 'ROC' else ('lower left' if split == 'C1Virus' else 'upper right')
        legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                           loc=location, title='AUROC' if curve == 'ROC' else 'PR area',
                           title_fontsize=6.8, fontsize=6.8, frameon=True,
                           facecolor='white', framealpha=.93, edgecolor='#DDDDDD',
                           handlelength=1.8, handletextpad=.45, labelspacing=.25, borderpad=.3)
        legend.get_frame().set_linewidth(.5)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT/f'five_fold_mean_four_panel.{ext}', dpi=600, facecolor='white')
    plt.close(fig)
    with (OUTPUT/'mean_curve_points.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.writer(f); writer.writerow(['split','curve','x','mean_y'])
        for (split, kind), vals in means.items():
            writer.writerows((split, kind, x, y) for x,y in zip(grid, vals))
    with (OUTPUT/'fold_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['fold','split','n','AUROC','PR_AUC_trapezoidal','AP']); w.writerows(fold_rows)
    with (OUTPUT/'baseline_corrected_coordinates.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['panel','method','x','y'])
        for (panel, method), xy in baseline.items():
            w.writerows((panel, method, x, y) for x,y in xy)
    report = {'summary': summaries, 'prediction_hashes': {':'.join(k):v for k,v in raw_curves.items()},
              'averaging': 'five fold curves linearly interpolated onto 1001 common FPR/recall points, then arithmetic mean',
              'legend': 'mean of five per-fold AUROC or trapezoidal PR-AUC; not necessarily area under plotted mean curve',
              'baseline': 'published curves are approximate raster digitizations, not raw predictions; published legend values',
              'yellow_repair': {'panel':'A', 'method':'TransPPI', 'removed_suspected_misclassified_pixels':removed,
                                'rendering':'remaining adjacent points connected linearly; corrected values are not raw observations'},
              'limitations': 'paper figure aggregation unspecified; historical C2Virus checkpoint selection; descriptive comparison only'}
    (OUTPUT/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Output:', OUTPUT)

if __name__ == '__main__':
    main()
