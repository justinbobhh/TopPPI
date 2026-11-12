import argparse
from pathlib import Path
import inspect

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BLUE = '#5960A5'
RED = '#B83F2B'

def tsne_project(features, seed, perplexity):
    if len(features) <= 4:
        raise ValueError('Need more than 4 real samples to draw a meaningful t-SNE plot.')
    n_features = min(50, features.shape[1], len(features) - 1)
    if n_features >= 2 and features.shape[1] > n_features:
        X = PCA(n_components=n_features, random_state=seed).fit_transform(features)
    else:
        X = features
    pp = min(float(perplexity), max(2.0, (len(features)-1)/3))
    params = dict(n_components=2, perplexity=pp, init='pca', learning_rate='auto', random_state=seed)
    argname = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
    params[argname] = 1100
    return TSNE(**params).fit_transform(X)

def main():
    p = argparse.ArgumentParser(description='TopPPI: project tsne.')
    p.add_argument('--panel', action='append', help='Optional custom PanelTitle=/path/to/features.npz; supply exactly six entries')
    p.add_argument('--input-dir', type=Path, default=Path('./viz_sixpanel'))
    p.add_argument('--output', default='./viz_sixpanel/structpeftcl_sixpanel_tsne.png')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--perplexity', type=float, default=30.0)
    p.add_argument('--max-per-class', type=int, default=0, help='0=use all samples; positive N=subsample at most N per label, deterministically')
    p.add_argument('--dpi', type=int, default=500)
    args = p.parse_args()
    if args.dpi < 100:
        p.error('Use a print-quality --dpi (300 or higher recommended).')

    if args.panel is None:
        names = ('Yeast', 'Multi-species_01', 'Multi-species_10',
                 'Multi-species_25', 'Multi-species_40', 'Multi-species_any')
        args.panel = [f"{name}={args.input_dir / (name.replace('-', '_') + '.npz')}"
                      for name in names]
    if len(args.panel) != 6:
        p.error('This figure requires exactly six panels: Yeast, _01, _10, _25, _40, _any')
    panels = []
    for entry in args.panel:
        if '=' not in entry:
            p.error(f'Invalid --panel {entry!r}: expected PanelTitle=path.npz')
        title, path = entry.split('=', 1)
        with np.load(path, allow_pickle=False) as data:
            X = data['features'].astype(np.float32)
            y = data['labels'].astype(np.int64)
        if len(y) != len(X) or X.ndim != 2 or X.shape[1] < 2:
            raise ValueError(f'{path}: features must be [N,D] with D>=2 and aligned labels')
        if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
            raise ValueError(f'{path}: features contain nonfinite values or labels are not binary')
        if (y == 0).sum() == 0 or (y == 1).sum() == 0:
            raise ValueError(f'{path}: cannot compare positive and negative classes with one class absent')
        if args.max_per_class > 0:
            rng = np.random.default_rng(args.seed)
            ix = []
            for cls in (0, 1):
                choices = np.flatnonzero(y == cls)
                ix.extend(rng.choice(choices, min(len(choices), args.max_per_class), replace=False).tolist())
            ix = np.sort(ix)
            X, y = X[ix], y[ix]
        points = tsne_project(X, args.seed, args.perplexity)
        panels.append((title, points, y))
        print(f'{title}: N={len(y)}, neg={(y==0).sum()}, pos={(y==1).sum()}')

    n = len(panels)
    cols = 3
    rows = 2
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
                         'svg.fonttype':'none', 'pdf.fonttype':42})
    fig = plt.figure(figsize=(12.6, 8.0), facecolor='white')
    outer = GridSpec(rows, cols, figure=fig, left=.045, right=.99, bottom=.085,
                     top=.985, wspace=.26, hspace=.34)

    for i, (title, xy, labels) in enumerate(panels):
        inner = outer[i // cols, i % cols].subgridspec(2,2, width_ratios=(4.8, .65),
                            height_ratios=(.6,4.7), wspace=.02, hspace=.02)
        ax = fig.add_subplot(inner[1,0])
        top = fig.add_subplot(inner[0,0], sharex=ax)
        right = fig.add_subplot(inner[1,1], sharey=ax)
        blank = fig.add_subplot(inner[0,1]);blank.set_axis_off()
        lowx, highx = xy[:,0].min(), xy[:,0].max()
        lowy, highy = xy[:,1].min(), xy[:,1].max()
        padx=max(1e-4,(highx-lowx)*.06);pady=max(1e-4,(highy-lowy)*.06)
        ax.set_xlim(lowx-padx, highx+padx);ax.set_ylim(lowy-pady, highy+pady)
        xbins = np.linspace(lowx-padx,highx+padx,24)
        ybins = np.linspace(lowy-pady,highy+pady,24)
        for label, color in ((0,BLUE),(1,RED)):
            sel = labels == label
            ax.scatter(xy[sel,0],xy[sel,1],s=2.0,c=color,alpha=.84,linewidths=0,rasterized=True)
            top.hist(xy[sel,0], bins=xbins, color=color, alpha=.92, rwidth=.88)
            right.hist(xy[sel,1],bins=ybins,color=color,alpha=.92,rwidth=.88,orientation='horizontal')
        ax.grid(True,linestyle=(0,(3,3)),linewidth=.52,color='#C6C6C6',zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both',which='both',labelbottom=False,labelleft=False,length=0)
        for side in ['left','right','bottom','top']:
            ax.spines[side].set_color('#8A8A8A');ax.spines[side].set_linewidth(.85)
        ax.set_xlabel(title,fontweight='bold',fontsize=12,labelpad=7)
        for h in (top,right):
            h.set_facecolor('white')
            h.tick_params(axis='both',which='both',length=0,labelbottom=False,labelleft=False)
            for s in h.spines.values(): s.set_visible(False)
        top.set_ylim(bottom=0);right.set_xlim(left=0)

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=args.dpi,facecolor='white',bbox_inches='tight')
    svg=out.with_suffix('.svg')
    fig.savefig(svg,facecolor='white',bbox_inches='tight')
    pdf=out.with_suffix('.pdf')
    fig.savefig(pdf,facecolor='white',bbox_inches='tight')
    plt.close(fig)
    print('Saved:', out, svg, 'and',pdf)

if __name__ == '__main__':
    main()
