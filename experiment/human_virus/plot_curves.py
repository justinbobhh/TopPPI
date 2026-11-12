import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/curves'
OUTPUT = ROOT.parents[1] / 'outputs/human_virus/figure2'
METHODS = ['TopPPI', 'DeepGNHV', 'HBFormer', 'TransPPI',
           'HVPPI', 'DeepViral', 'CAPHV']
COLORS = ['#3C787B', '#D95459', '#E59859', '#D6BE28',
          '#68A99A', '#77A1C8', '#8773A6']
PANELS = [('A', 'C1Virus', 'ROC'), ('B', 'C1Virus', 'PR'),
          ('C', 'C2Virus', 'ROC'), ('D', 'C2Virus', 'PR')]

def smooth_display(x, y, kind):

    order = np.argsort(x, kind='stable')
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    unique_y = np.bincount(inverse, weights=y) / np.bincount(inverse)
    if len(unique_x) < 3:
        raise ValueError('At least three distinct x coordinates are required')
    start, end = float(y[0]), float(y[-1])
    increasing = kind == 'ROC'
    if (increasing and start > end) or (not increasing and start < end):
        raise ValueError('Endpoints incompatible with requested monotonicity')
    if increasing:
        projected = IsotonicRegression(
            increasing=True, y_min=start, y_max=end
        ).fit_transform(unique_x, unique_y)
    else:
        projected = np.maximum.accumulate(unique_y[::-1])[::-1]
        projected = np.clip(projected, end, start)
    projected[0], projected[-1] = start, end

    trend_x = np.linspace(unique_x[0], unique_x[-1], 2001)
    trend_y = gaussian_filter1d(
        np.interp(trend_x, unique_x, projected), sigma=.018 * 2000,
        mode='nearest'
    )

    unit_knots = (1 - np.cos(np.linspace(0, np.pi, 25))) / 2
    knots = unique_x[0] + unit_knots * (unique_x[-1] - unique_x[0])
    values = np.interp(knots, trend_x, trend_y)
    values = np.clip(values, min(start, end), max(start, end))
    values[0], values[-1] = start, end
    dense_x = np.unique(np.r_[np.linspace(unique_x[0], unique_x[-1], 2001), knots])
    dense_y = np.clip(PchipInterpolator(knots, values)(dense_x), 0, 1)
    dense_y[0], dense_y[-1] = start, end
    delta = np.diff(dense_y)
    if (increasing and np.any(delta < -1e-12)) or (not increasing and np.any(delta > 1e-12)):
        raise AssertionError('Display curve failed monotonicity check')
    raw_on_dense = np.interp(dense_x, unique_x, unique_y)
    audit = {'raw_coordinate_area': float(np.trapz(unique_y, unique_x)),
             'display_area': float(np.trapz(dense_y, dense_x)),
             'max_vertical_change': float(np.max(np.abs(dense_y - raw_on_dense))),
             'anchor_count': len(knots),
             'start_preserved': bool(dense_y[0] == start),
             'end_preserved': bool(dense_y[-1] == end)}
    audit['display_area_minus_raw'] = audit['display_area'] - audit['raw_coordinate_area']
    return dense_x, dense_y, audit

def main():
    global SOURCE, OUTPUT
    parser = argparse.ArgumentParser(description='TopPPI: plot curves.')
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    SOURCE, OUTPUT = args.source_dir.resolve(), args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    verified = json.loads((SOURCE / 'verification.json').read_text(encoding='utf-8'))
    curves, numbers = {}, {}
    with (SOURCE / 'baseline_corrected_coordinates.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            curves.setdefault((r['panel'], r['method']), []).append((float(r['x']), float(r['y'])))
    with (SOURCE / 'mean_curve_points.csv').open(encoding='utf-8', newline='') as f:
        panel_map = {(s, c): a for a, s, c in PANELS}
        for r in csv.DictReader(f):
            curves.setdefault((panel_map[(r['split'], r['curve'])], METHODS[0]), []).append((float(r['x']), float(r['mean_y'])))
    with (ROOT / 'figure_reference' / 'published_figure2_metrics.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            numbers[(r['panel'], r['method'])] = float(r['reported_area'])
    for letter, split, kind in PANELS:
        key = 'mean_fold_AUROC' if kind == 'ROC' else 'mean_fold_PR_AUC_trapezoidal'
        numbers[(letter, METHODS[0])] = verified['summary'][split][key]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 9, 'axes.labelweight': 'normal',
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'path.simplify': False, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.3))
    fig.subplots_adjust(left=.105, right=.98, bottom=.08, top=.96, wspace=.28, hspace=.30)
    audit, display_rows = {}, []
    for ax, (letter, split, kind) in zip(axes.flat, PANELS):
        handles = {}
        for i in [1, 2, 3, 4, 5, 6, 0]:
            method = METHODS[i]
            xy = np.asarray(curves[(letter, method)])
            x, y, changes = smooth_display(xy[:, 0], xy[:, 1], kind)
            audit[f'{letter}:{method}'] = changes
            display_rows.extend((letter, method, a, b) for a, b in zip(x, y))
            if method == METHODS[0] and kind == 'ROC':
                x, y = np.r_[0, x], np.r_[0, y]
            handles[method], = ax.plot(
                x, y, color=COLORS[i], lw=1.5 if i == 0 else 1.2,
                linestyle='-', alpha=.95, antialiased=True,
                solid_capstyle='round', solid_joinstyle='round', zorder=3 if i == 0 else 2,
                label=f'{method} ({numbers[(letter, method)]:.3f})')
        ax.set(xlim=(0, 1), ylim=(0, 1),
               xlabel='False positive rate' if kind == 'ROC' else 'Recall',
               ylabel='True positive rate' if kind == 'ROC' else 'Precision')
        ax.set_box_aspect(1/1.06)
        ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6))
        ax.grid(color='#D9D9D9', linewidth=.5, alpha=.7); ax.set_axisbelow(True)
        ax.tick_params(width=.7, length=3)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(.8)
        ax.text(-.15, 1.03, letter, transform=ax.transAxes, fontsize=13, va='bottom')
        loc = 'lower right' if kind == 'ROC' else ('lower left' if split == 'C1Virus' else 'upper right')
        legend = ax.legend([handles[m] for m in METHODS],
                           [handles[m].get_label() for m in METHODS], loc=loc,
                           title='AUROC' if kind == 'ROC' else 'PR area',
                           title_fontsize=6.8, fontsize=6.8, frameon=True,
                           facecolor='white', framealpha=.93, edgecolor='#DDDDDD',
                           handlelength=1.8, handletextpad=.45, labelspacing=.25, borderpad=.3)
        legend.get_frame().set_linewidth(.5)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT/f'trend_four_panel.{ext}', dpi=600, facecolor='white')
    plt.close(fig)
    with (OUTPUT/'display_curve_points.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.writer(f); w.writerow(['panel','method','x','display_y']); w.writerows(display_rows)
    report = {'legend_metrics': {':'.join(k):v for k,v in numbers.items()},
              'display_processing': 'ROC isotonic projection; PR upper nonincreasing envelope; shared Gaussian low-frequency scale 0.018 x-span; 25 common cosine-spaced anchors; PCHIP',
              'interpretation': 'display smoothing is not a new empirical curve; legend metrics are unchanged original fold means/published values; use raw data for threshold analysis',
              'source_limits': verified['limitations'], 'curve_audit': audit}
    report['coordinate_area_ranking_preserved'] = {}
    for panel, _, _ in PANELS:
        keys = [f'{panel}:{m}' for m in METHODS]
        before = sorted(keys, key=lambda k: audit[k]['raw_coordinate_area'], reverse=True)
        after = sorted(keys, key=lambda k: audit[k]['display_area'], reverse=True)
        report['coordinate_area_ranking_preserved'][panel] = before == after
    (OUTPUT/'smoothing_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved to', OUTPUT)
    print('Largest absolute display area change:', max(abs(a['display_area_minus_raw']) for a in audit.values()))

if __name__ == '__main__':
    main()
