import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

root = Path(__file__).resolve().parent
source = root / 'results'
out = root.parents[1] / 'outputs/visualization'
out.mkdir(parents=True, exist_ok=True)
panels = ['Yeast', 'Multi-Species_01', 'Multi-Species_10',
          'Multi-Species_25', 'Multi-Species_40', 'Multi-Species_any']
values = {name: [] for name in panels}
with (source / 'tsne_coordinates.csv').open(encoding='utf-8', newline='') as handle:
    for r in csv.DictReader(handle):
        values[r['panel']].append((float(r['tsne_1']), float(r['tsne_2']), int(r['label'])))

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(10.8, 6.9), facecolor='white')
outer = GridSpec(2, 3, figure=fig, left=.045, right=.99, bottom=.075, top=.98,
                 wspace=.16, hspace=.17)
colors = {0: '#59619E', 1: '#AD3829'}
for i, name in enumerate(panels):
    arr = np.asarray(values[name]); xy = arr[:, :2]; labels = arr[:, 2].astype(int)
    grid = outer[i // 3, i % 3].subgridspec(2, 2, width_ratios=(1, .14),
                  height_ratios=(.13, 1), wspace=.01, hspace=.01)
    ax = fig.add_subplot(grid[1, 0]); top = fig.add_subplot(grid[0, 0], sharex=ax)
    side = fig.add_subplot(grid[1, 1], sharey=ax)
    low = xy.min(axis=0); high = xy.max(axis=0); pad = (high-low)*.06
    ax.set_xlim(low[0]-pad[0], high[0]+pad[0]); ax.set_ylim(low[1]-pad[1], high[1]+pad[1])
    bins_x = np.linspace(low[0]-pad[0], high[0]+pad[0], 14)
    bins_y = np.linspace(low[1]-pad[1], high[1]+pad[1], 14)
    for label in (0, 1):
        pts = xy[labels == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=2.8, color=colors[label], alpha=.88,
                   linewidths=0, rasterized=True)
        top.hist(pts[:, 0], bins=bins_x, color=colors[label], alpha=.92, rwidth=.58)
        side.hist(pts[:, 1], bins=bins_y, color=colors[label], alpha=.92,
                  rwidth=.58, orientation='horizontal')
    ax.grid(True, linestyle=(0, (2.3, 2.7)), linewidth=.48, color='#bdbdbd')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values(): spine.set_color('#929292'); spine.set_linewidth(.75)
    ax.set_xlabel(name, fontweight='bold', fontsize=13, labelpad=5)
    for marginal in (top, side):
        marginal.tick_params(axis='both', which='both', bottom=False, left=False,
                             labelbottom=False, labelleft=False)
        for spine in marginal.spines.values(): spine.set_visible(False)
    top.set_ylim(bottom=0); side.set_xlim(left=0)

for ext in ('png', 'pdf', 'svg'):
    fig.savefig(out / ('figure5_tsne.' + ext), dpi=500,
                facecolor='white', bbox_inches='tight')
plt.close(fig)
